"""
Backtesting module for trading strategy evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class Backtester:
    """Backtest trading strategies based on model predictions."""
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate per trade (e.g., 0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.results = None
        
    def run_simple_strategy(self, dates: np.ndarray, actual_prices: np.ndarray,
                          predicted_prices: np.ndarray, 
                          strategy: str = 'long_short') -> Dict:
        """
        Run a simple trading strategy backtest.
        
        Args:
            dates: Array of dates
            actual_prices: Actual price series
            predicted_prices: Predicted price series
            strategy: Strategy type ('long_only', 'long_short', 'threshold')
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running {strategy} strategy backtest...")
        
        portfolio_value = [self.initial_capital]
        cash = self.initial_capital
        shares = 0
        trades = []
        
        for i in range(1, len(actual_prices)):
            prev_price = actual_prices[i-1]
            curr_price = actual_prices[i]
            prev_pred = predicted_prices[i-1]
            curr_pred = predicted_prices[i]
            
            # Determine signal
            if strategy == 'long_only':
                signal = 1 if curr_pred > prev_pred else 0
            elif strategy == 'long_short':
                signal = 1 if curr_pred > prev_pred else -1
            elif strategy == 'threshold':
                threshold = 0.01  # 1% threshold
                pred_change = (curr_pred - prev_pred) / prev_pred
                if pred_change > threshold:
                    signal = 1
                elif pred_change < -threshold:
                    signal = -1
                else:
                    signal = 0
            else:
                signal = 0
            
            # Execute trades
            if signal == 1 and shares <= 0:
                # Buy signal
                if shares < 0:
                    # Cover short position
                    cash -= shares * curr_price * (1 + self.commission)
                    trades.append({
                        'date': dates[i],
                        'action': 'cover',
                        'price': curr_price,
                        'shares': -shares
                    })
                    shares = 0
                
                # Go long
                shares_to_buy = cash / (curr_price * (1 + self.commission))
                if shares_to_buy > 0:
                    cash -= shares_to_buy * curr_price * (1 + self.commission)
                    shares += shares_to_buy
                    trades.append({
                        'date': dates[i],
                        'action': 'buy',
                        'price': curr_price,
                        'shares': shares_to_buy
                    })
            
            elif signal == -1 and shares >= 0:
                # Sell/Short signal
                if shares > 0:
                    # Sell long position
                    cash += shares * curr_price * (1 - self.commission)
                    trades.append({
                        'date': dates[i],
                        'action': 'sell',
                        'price': curr_price,
                        'shares': shares
                    })
                    shares = 0
                
                # Go short (if long_short strategy)
                if strategy == 'long_short':
                    shares_to_short = cash / (curr_price * (1 + self.commission))
                    if shares_to_short > 0:
                        cash += shares_to_short * curr_price * (1 - self.commission)
                        shares -= shares_to_short
                        trades.append({
                            'date': dates[i],
                            'action': 'short',
                            'price': curr_price,
                            'shares': shares_to_short
                        })
            
            # Calculate portfolio value
            current_value = cash + shares * curr_price
            portfolio_value.append(current_value)
        
        # Close any remaining positions
        if shares != 0:
            cash += shares * actual_prices[-1] * (1 - self.commission)
            trades.append({
                'date': dates[-1],
                'action': 'close',
                'price': actual_prices[-1],
                'shares': abs(shares)
            })
        
        final_value = cash
        portfolio_value[-1] = final_value
        
        # Calculate metrics
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        portfolio_returns = np.diff(portfolio_value) / portfolio_value[:-1]
        sharpe_ratio = np.sqrt(252) * np.mean(portfolio_returns) / np.std(portfolio_returns) if np.std(portfolio_returns) > 0 else 0
        
        max_drawdown = self._calculate_max_drawdown(np.array(portfolio_value))
        
        results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'num_trades': len(trades),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_values': portfolio_value,
            'trades': trades
        }
        
        # Buy and hold comparison
        buy_hold_shares = self.initial_capital / actual_prices[0]
        buy_hold_final = buy_hold_shares * actual_prices[-1]
        buy_hold_return = ((buy_hold_final - self.initial_capital) / self.initial_capital) * 100
        
        results['buy_hold_return'] = buy_hold_return
        results['excess_return'] = total_return - buy_hold_return
        
        self.results = results
        
        logger.info(f"Backtest completed: Total Return={total_return:.2f}%, Sharpe={sharpe_ratio:.2f}, Max DD={max_drawdown:.2f}%")
        
        return results
    
    def run_threshold_strategy(self, dates: np.ndarray, actual_prices: np.ndarray,
                              predicted_prices: np.ndarray,
                              buy_threshold: float = 0.02,
                              sell_threshold: float = -0.01) -> Dict:
        """
        Run threshold-based strategy.
        
        Args:
            dates: Array of dates
            actual_prices: Actual price series
            predicted_prices: Predicted price series
            buy_threshold: Minimum predicted return to buy
            sell_threshold: Maximum predicted return to sell
            
        Returns:
            Dictionary with backtest results
        """
        portfolio_value = [self.initial_capital]
        cash = self.initial_capital
        shares = 0
        trades = []
        
        for i in range(1, len(actual_prices)):
            curr_price = actual_prices[i]
            pred_return = (predicted_prices[i] - actual_prices[i-1]) / actual_prices[i-1]
            
            # Buy signal
            if pred_return > buy_threshold and shares == 0:
                shares = cash / (curr_price * (1 + self.commission))
                cash = 0
                trades.append({
                    'date': dates[i],
                    'action': 'buy',
                    'price': curr_price,
                    'shares': shares,
                    'predicted_return': pred_return
                })
            
            # Sell signal
            elif pred_return < sell_threshold and shares > 0:
                cash = shares * curr_price * (1 - self.commission)
                trades.append({
                    'date': dates[i],
                    'action': 'sell',
                    'price': curr_price,
                    'shares': shares,
                    'predicted_return': pred_return
                })
                shares = 0
            
            current_value = cash + shares * curr_price
            portfolio_value.append(current_value)
        
        # Close position if needed
        if shares > 0:
            cash = shares * actual_prices[-1] * (1 - self.commission)
            shares = 0
        
        final_value = cash
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'num_trades': len(trades),
            'portfolio_values': portfolio_value,
            'trades': trades
        }
    
    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cumulative_max) / cumulative_max
        return np.min(drawdown) * 100
    
    def get_trade_statistics(self) -> Dict:
        """Calculate statistics about trades."""
        if self.results is None or 'trades' not in self.results:
            return {}
        
        trades = pd.DataFrame(self.results['trades'])
        
        if len(trades) == 0:
            return {'num_trades': 0}
        
        buy_trades = trades[trades['action'].isin(['buy', 'short'])]
        sell_trades = trades[trades['action'].isin(['sell', 'cover', 'close'])]
        
        stats = {
            'num_trades': len(trades),
            'num_buy': len(buy_trades),
            'num_sell': len(sell_trades),
            'avg_buy_price': buy_trades['price'].mean() if len(buy_trades) > 0 else 0,
            'avg_sell_price': sell_trades['price'].mean() if len(sell_trades) > 0 else 0,
        }
        
        return stats
    
    def create_performance_report(self) -> pd.DataFrame:
        """Create a formatted performance report."""
        if self.results is None:
            logger.warning("No backtest results available")
            return pd.DataFrame()
        
        report_data = []
        
        metrics = [
            ('Initial Capital', f"${self.results['initial_capital']:,.2f}"),
            ('Final Value', f"${self.results['final_value']:,.2f}"),
            ('Total Return', f"{self.results['total_return']:.2f}%"),
            ('Buy & Hold Return', f"{self.results.get('buy_hold_return', 0):.2f}%"),
            ('Excess Return', f"{self.results.get('excess_return', 0):.2f}%"),
            ('Number of Trades', str(self.results['num_trades'])),
            ('Sharpe Ratio', f"{self.results.get('sharpe_ratio', 0):.2f}"),
            ('Max Drawdown', f"{self.results.get('max_drawdown', 0):.2f}%"),
        ]
        
        for metric, value in metrics:
            report_data.append({'Metric': metric, 'Value': value})
        
        return pd.DataFrame(report_data)
