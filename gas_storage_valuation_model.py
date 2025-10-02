import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

class EnhancedGasStorageModel:

    def __init__(self, max_storage_volume: float, injection_rate: float,
                 withdrawal_rate: float, storage_cost_per_unit: float):
        self.max_storage_volume = max_storage_volume
        self.injection_rate = injection_rate
        self.withdrawal_rate = withdrawal_rate
        self.storage_cost_per_unit = storage_cost_per_unit

    def calculate_contract_value(self,
                                injection_dates: List[str],
                                withdrawal_dates: List[str],
                                injection_prices: List[float],
                                withdrawal_prices: List[float],
                                verbose: bool = False) -> Dict:
        
        # Convert dates to datetime
        inj_dates = pd.to_datetime(injection_dates)
        wd_dates = pd.to_datetime(withdrawal_dates)
        inj_prices = np.array(injection_prices)
        wd_prices = np.array(withdrawal_prices)

        # Assume equal volumes for simplicity
        inj_volumes = np.full(len(inj_dates), self.injection_rate)
        wd_volumes = np.full(len(wd_dates), self.withdrawal_rate)

        # Build transactions DataFrame
        transactions = []
        storage_level = 0.0
        utilization_pct = []
        cash_flows = []
        for i in range(max(len(inj_dates), len(wd_dates))):
            # Injection
            if i < len(inj_dates):
                storage_level += inj_volumes[i]
                storage_level = min(storage_level, self.max_storage_volume)
                cash_flow = -inj_volumes[i] * inj_prices[i] - inj_volumes[i] * self.storage_cost_per_unit
                transactions.append({
                    'Date': inj_dates[i],
                    'Type': 'Injection',
                    'Volume_MMBtu': inj_volumes[i],
                    'Price_per_MMBtu': inj_prices[i],
                    'Storage_Level': storage_level,
                    'Cash_Flow': cash_flow,
                    'Utilization_Pct': storage_level / self.max_storage_volume * 100
                })
                cash_flows.append(cash_flow)
                utilization_pct.append(storage_level / self.max_storage_volume * 100)
            # Withdrawal
            if i < len(wd_dates):
                storage_level -= wd_volumes[i]
                storage_level = max(storage_level, 0.0)
                cash_flow = wd_volumes[i] * wd_prices[i]
                transactions.append({
                    'Date': wd_dates[i],
                    'Type': 'Withdrawal',
                    'Volume_MMBtu': wd_volumes[i],
                    'Price_per_MMBtu': wd_prices[i],
                    'Storage_Level': storage_level,
                    'Cash_Flow': cash_flow,
                    'Utilization_Pct': storage_level / self.max_storage_volume * 100
                })
                cash_flows.append(cash_flow)
                utilization_pct.append(storage_level / self.max_storage_volume * 100)

        transactions_df = pd.DataFrame(transactions)
        transactions_df.sort_values('Date', inplace=True)
        transactions_df.reset_index(drop=True, inplace=True)

        # Performance metrics
        gross_margin = transactions_df[transactions_df['Type'] == 'Withdrawal']['Cash_Flow'].sum()
        net_margin = gross_margin + transactions_df[transactions_df['Type'] == 'Injection']['Cash_Flow'].sum()
        storage_cost_total = self.storage_cost_per_unit * transactions_df['Volume_MMBtu'].sum()
        roi = (net_margin / abs(transactions_df[transactions_df['Type'] == 'Injection']['Cash_Flow'].sum())) * 100 if abs(transactions_df[transactions_df['Type'] == 'Injection']['Cash_Flow'].sum()) > 0 else 0
        storage_cost_ratio = (storage_cost_total / gross_margin * 100) if gross_margin > 0 else 0
        utilization_rate = np.mean(utilization_pct) / 100 if utilization_pct else 0

        results = {
            'transactions': transactions_df,
            'net_contract_value': net_margin,
            'performance_metrics': {
                'gross_margin': gross_margin,
                'net_margin': net_margin,
                'return_on_investment': roi,
                'storage_cost_ratio': storage_cost_ratio
            },
            'utilization_rate': utilization_rate
        }
        if verbose:
            print(transactions_df)
            print("Performance Metrics:", results['performance_metrics'])
        return results

    def scenario_analysis(self, base_scenario: Dict, sensitivity_params: Dict) -> pd.DataFrame:
        
        results = []
        for param, values in sensitivity_params.items():
            for val in values:
                # Set parameter
                if param == 'storage_cost':
                    self.storage_cost_per_unit = val
                elif param == 'capacity':
                    self.max_storage_volume = val
                # Calculate contract value
                res = self.calculate_contract_value(
                    base_scenario['injection_dates'],
                    base_scenario['withdrawal_dates'],
                    base_scenario['injection_prices'],
                    base_scenario['withdrawal_prices'],
                    verbose=False
                )
                results.append({
                    'parameter': param,
                    'value': val,
                    'net_value': res['net_contract_value'],
                    'roi': res['performance_metrics']['return_on_investment'],
                    'utilization': res['utilization_rate']
                })
        return pd.DataFrame(results)

    def plot_comprehensive_analysis(self, results: Dict, save_plots: bool = False):
       
        transactions = results['transactions']
        if transactions.empty:
            print("No transactions to plot!")
            return

        transactions['Date'] = pd.to_datetime(transactions['Date'])

        fig = plt.figure(figsize=(20, 15))

        # Plot 1: Cash Flow Timeline
        plt.subplot(3, 3, 1)
        colors = ['red' if cf < 0 else 'green' for cf in transactions['Cash_Flow']]
        plt.bar(transactions['Date'], transactions['Cash_Flow'], color=colors, alpha=0.7)
        plt.title('Cash Flow Timeline', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Cash Flow ($)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        for i, (date, cf) in enumerate(zip(transactions['Date'], transactions['Cash_Flow'])):
            plt.text(date, cf + (abs(cf) * 0.05 if cf > 0 else -abs(cf) * 0.05),
                     f'${cf:,.0f}', ha='center', va='bottom' if cf > 0 else 'top', fontsize=8)

        # Plot 2: Storage Level Over Time
        plt.subplot(3, 3, 2)
        plt.plot(transactions['Date'], transactions['Storage_Level'],
                 marker='o', linewidth=3, markersize=8, color='blue')
        plt.fill_between(transactions['Date'], transactions['Storage_Level'],
                         alpha=0.3, color='lightblue')
        plt.axhline(y=self.max_storage_volume, color='red', linestyle='--',
                    label=f'Max Capacity: {self.max_storage_volume:,.0f}')
        plt.title('Storage Level Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Storage Level (MMBtu)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 3: Price Comparison (Injection vs Withdrawal)
        plt.subplot(3, 3, 3)
        inj_data = transactions[transactions['Type'] == 'Injection']
        with_data = transactions[transactions['Type'] == 'Withdrawal']
        x_pos = np.arange(len(transactions))
        if not inj_data.empty:
            inj_positions = [i for i, t in enumerate(transactions['Type']) if t == 'Injection']
            plt.bar([x_pos[i] for i in inj_positions],
                    [transactions.iloc[i]['Price_per_MMBtu'] for i in inj_positions],
                    width=0.4, label='Injection Prices', color='red', alpha=0.7)
        if not with_data.empty:
            with_positions = [i for i, t in enumerate(transactions['Type']) if t == 'Withdrawal']
            plt.bar([x_pos[i] + 0.4 for i in with_positions],
                    [transactions.iloc[i]['Price_per_MMBtu'] for i in with_positions],
                    width=0.4, label='Withdrawal Prices', color='green', alpha=0.7)
        plt.title('Price Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Transaction Order')
        plt.ylabel('Price ($/MMBtu)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 4: Volume Analysis
        plt.subplot(3, 3, 4)
        injection_vols = inj_data['Volume_MMBtu'].tolist() if not inj_data.empty else [0]
        withdrawal_vols = with_data['Volume_MMBtu'].tolist() if not with_data.empty else [0]
        categories = ['Injections', 'Withdrawals']
        volumes = [sum(injection_vols), sum(withdrawal_vols)]
        colors = ['red', 'green']
        bars = plt.bar(categories, volumes, color=colors, alpha=0.7)
        plt.title('Total Volume Analysis', fontsize=14, fontweight='bold')
        plt.ylabel('Volume (MMBtu)')
        for bar, vol in zip(bars, volumes):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(volumes)*0.01,
                     f'{vol:,.0f}', ha='center', va='bottom', fontweight='bold')
        plt.grid(True, alpha=0.3)

        # Plot 5: Utilization Rate
        plt.subplot(3, 3, 5)
        utilization_pct = transactions['Utilization_Pct']
        plt.plot(transactions['Date'], utilization_pct,
                 marker='s', linewidth=2, markersize=6, color='purple')
        plt.fill_between(transactions['Date'], utilization_pct, alpha=0.3, color='lavender')
        plt.title('Storage Utilization Rate', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Utilization (%)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # Plot 6: Cumulative P&L
        plt.subplot(3, 3, 6)
        cumulative_pnl = transactions['Cash_Flow'].cumsum()
        plt.plot(transactions['Date'], cumulative_pnl, marker='o', linewidth=3, color='darkgreen')
        plt.fill_between(transactions['Date'], cumulative_pnl, alpha=0.3, color='lightgreen')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.title('Cumulative P&L', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Cumulative P&L ($)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # Plot 7: Performance Metrics Dashboard
        plt.subplot(3, 3, 7)
        metrics = results['performance_metrics']
        metric_names = ['ROI (%)', 'Gross Margin ($000)', 'Net Margin ($000)', 'Storage Cost Ratio (%)']
        metric_values = [
            metrics['return_on_investment'],
            metrics['gross_margin'] / 1000,
            metrics['net_margin'] / 1000,
            metrics['storage_cost_ratio']
        ]
        colors = ['green' if val > 0 else 'red' for val in metric_values]
        bars = plt.barh(metric_names, metric_values, color=colors, alpha=0.7)
        for i, (bar, val) in enumerate(zip(bars, metric_values)):
            plt.text(val + max(abs(v) for v in metric_values)*0.02, bar.get_y() + bar.get_height()/2,
                     f'{val:.1f}', ha='left' if val > 0 else 'right', va='center', fontweight='bold')
        plt.title('Performance Metrics', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')

        # Plot 8: Transaction Type Distribution (Pie Chart)
        plt.subplot(3, 3, 8)
        type_counts = transactions['Type'].value_counts()
        colors = ['red', 'green']
        plt.pie(type_counts.values, labels=type_counts.index,
                autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Transaction Distribution', fontsize=14, fontweight='bold')

        # Plot 9: Price Spread Analysis
        plt.subplot(3, 3, 9)
        if not inj_data.empty and not with_data.empty:
            avg_inj_price = inj_data['Price_per_MMBtu'].mean()
            avg_with_price = with_data['Price_per_MMBtu'].mean()
            spread = avg_with_price - avg_inj_price
            categories = ['Avg Injection\nPrice', 'Avg Withdrawal\nPrice', 'Price Spread']
            values = [avg_inj_price, avg_with_price, spread]
            colors = ['red', 'green', 'blue']
            bars = plt.bar(categories, values, color=colors, alpha=0.7)
            for bar, val in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                         f'${val:.2f}', ha='center', va='bottom', fontweight='bold')
            plt.title('Price Spread Analysis', fontsize=14, fontweight='bold')
            plt.ylabel('Price ($/MMBtu)')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Insufficient Data\nfor Spread Analysis',
                     ha='center', va='center', transform=plt.gca().transAxes,
                     fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            plt.title('Price Spread Analysis', fontsize=14, fontweight='bold')

        fig.suptitle(f'🏭 Gas Storage Contract Analysis Dashboard\n'
                     f'Net Value: ${results["net_contract_value"]:,.2f} | '
                     f'ROI: {metrics["return_on_investment"]:.1f}% | '
                     f'Utilization: {results["utilization_rate"]:.1%}',
                     fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_plots:
            plt.savefig('gas_storage_analysis.png', dpi=300, bbox_inches='tight')
            print("Plot saved as 'gas_storage_analysis.png'")

        plt.show()
        return fig

    def plot_scenario_comparison(self, scenario_results: pd.DataFrame, save_plots: bool = False):
        """
        Plot scenario analysis comparison charts
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        for param in scenario_results['parameter'].unique():
            data = scenario_results[scenario_results['parameter'] == param]
            axes[0,0].plot(data['value'], data['net_value'], marker='o', label=param, linewidth=2, markersize=6)
        axes[0,0].set_title('Net Value Sensitivity', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Parameter Value')
        axes[0,0].set_ylabel('Net Value ($)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        for param in scenario_results['parameter'].unique():
            data = scenario_results[scenario_results['parameter'] == param]
            axes[0,1].plot(data['value'], data['roi'], marker='s', label=param, linewidth=2, markersize=6)
        axes[0,1].set_title('ROI Sensitivity', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Parameter Value')
        axes[0,1].set_ylabel('ROI (%)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        for param in scenario_results['parameter'].unique():
            data = scenario_results[scenario_results['parameter'] == param]
            axes[1,0].plot(data['value'], data['utilization'], marker='^', label=param, linewidth=2, markersize=6)
        axes[1,0].set_title('⚡ Utilization Sensitivity', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Parameter Value')
        axes[1,0].set_ylabel('Utilization Rate')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        if len(scenario_results['parameter'].unique()) > 1:
            colors = plt.cm.Set3(np.linspace(0, 1, len(scenario_results)))
            scatter = axes[1,1].scatter(scenario_results['net_value'], scenario_results['roi'],
                                        s=scenario_results['utilization']*1000,
                                        c=colors, alpha=0.6)
            axes[1,1].set_title('Multi-Parameter Analysis', fontsize=14, fontweight='bold')
            axes[1,1].set_xlabel('Net Value ($)')
            axes[1,1].set_ylabel('ROI (%)')
            for i, row in scenario_results.iterrows():
                axes[1,1].annotate(f"{row['parameter']}\n{row['value']}",
                                   (row['net_value'], row['roi']),
                                   xytext=(5, 5), textcoords='offset points', fontsize=8)
        else:
            axes[1,1].text(0.5, 0.5, 'Single Parameter\nAnalysis', ha='center', va='center',
                           transform=axes[1,1].transAxes, fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            axes[1,1].set_title('Multi-Parameter Analysis', fontsize=14, fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        plt.suptitle('Scenario Analysis Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_plots:
            plt.savefig('scenario_analysis.png', dpi=300, bbox_inches='tight')
            print("Plot saved as 'scenario_analysis.png'")
        plt.show()
        return fig

    def plot_historical_price_analysis(self, price_data: pd.DataFrame, results: Dict = None, save_plots: bool = False):
        """
        Plot historical price data with injection/withdrawal opportunities
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        price_data_copy = price_data.copy()
        price_data_copy['Dates'] = pd.to_datetime(price_data_copy['Dates'])
        axes[0,0].plot(price_data_copy['Dates'], price_data_copy['Prices'],
                       linewidth=2, color='blue', alpha=0.8)
        axes[0,0].fill_between(price_data_copy['Dates'], price_data_copy['Prices'],
                              alpha=0.3, color='lightblue')
        if results and 'optimization_strategy' in results:
            strategy = results['optimization_strategy']
            inj_threshold = strategy['injection_price_threshold']
            with_threshold = strategy['withdrawal_price_threshold']
            axes[0,0].axhline(y=inj_threshold, color='red', linestyle='--',
                              label=f'Injection Threshold: ${inj_threshold:.2f}')
            axes[0,0].axhline(y=with_threshold, color='green', linestyle='--',
                              label=f'Withdrawal Threshold: ${with_threshold:.2f}')
            axes[0,0].legend()
        axes[0,0].set_title('Historical Natural Gas Prices', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Date')
        axes[0,0].set_ylabel('Price ($/MMBtu)')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,1].hist(price_data_copy['Prices'], bins=15, color='skyblue', alpha=0.7, edgecolor='black')
        axes[0,1].axvline(price_data_copy['Prices'].mean(), color='red', linestyle='--',
                          label=f'Mean: ${price_data_copy["Prices"].mean():.2f}')
        axes[0,1].axvline(price_data_copy['Prices'].median(), color='green', linestyle='--',
                          label=f'Median: ${price_data_copy["Prices"].median():.2f}')
        axes[0,1].set_title('Price Distribution', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Price ($/MMBtu)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        window = min(12, len(price_data_copy) // 4)
        rolling_std = price_data_copy['Prices'].rolling(window=window).std()
        axes[1,0].plot(price_data_copy['Dates'], rolling_std, color='purple', linewidth=2)
        axes[1,0].fill_between(price_data_copy['Dates'], rolling_std, alpha=0.3, color='lavender')
        axes[1,0].set_title(f'Price Volatility (Rolling {window}-Month Std Dev)', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Date')
        axes[1,0].set_ylabel('Price Volatility')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].tick_params(axis='x', rotation=45)
        price_changes = price_data_copy['Prices'].pct_change().dropna() * 100
        colors = ['red' if x < 0 else 'green' for x in price_changes]
        axes[1,1].bar(range(len(price_changes)), price_changes, color=colors, alpha=0.7)
        axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.8)
        axes[1,1].set_title('Month-over-Month Price Changes', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Time Period')
        axes[1,1].set_ylabel('Price Change (%)')
        axes[1,1].grid(True, alpha=0.3)
        plt.suptitle('Historical Price Analysis Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_plots:
            plt.savefig('historical_price_analysis.png', dpi=300, bbox_inches='tight')
            print("Plot saved as 'historical_price_analysis.png'")
        plt.show()
        return fig

def run_complete_analysis_with_plots():
    """
    Run complete analysis with all visualization options
    """
    print(" RUNNING COMPLETE ANALYSIS WITH VISUALIZATIONS")
    print("="*60)
    model = EnhancedGasStorageModel(
        max_storage_volume=50000,
        injection_rate=2500,
        withdrawal_rate=2500,
        storage_cost_per_unit=0.005
    )
    injection_dates = ['2024-01-15', '2024-02-15', '2024-03-15']
    withdrawal_dates = ['2024-07-15', '2024-08-15', '2024-09-15']
    injection_prices = [10.50, 10.20, 10.30]
    withdrawal_prices = [12.50, 12.80, 12.20]
    results = model.calculate_contract_value(
        injection_dates=injection_dates,
        withdrawal_dates=withdrawal_dates,
        injection_prices=injection_prices,
        withdrawal_prices=withdrawal_prices,
        verbose=True
    )
    print(f"\n RESULTS SUMMARY:")
    print(f"Net Contract Value: ${results['net_contract_value']:,.2f}")
    print(f"ROI: {results['performance_metrics']['return_on_investment']:.1f}%")
    print("\n Generating comprehensive analysis plots...")
    model.plot_comprehensive_analysis(results, save_plots=True)
    print("\n Running scenario analysis...")
    base_scenario = {
        'injection_dates': injection_dates,
        'withdrawal_dates': withdrawal_dates,
        'injection_prices': injection_prices,
        'withdrawal_prices': withdrawal_prices
    }
    sensitivity_params = {
        'storage_cost': [0.001, 0.005, 0.01, 0.02],
        'capacity': [25000, 50000, 75000, 100000]
    }
    scenario_results = model.scenario_analysis(base_scenario, sensitivity_params)
    model.plot_scenario_comparison(scenario_results, save_plots=True)
    try:
        df = pd.read_csv('Nat_Gas.csv')
        if 'Dates' not in df.columns or 'Prices' not in df.columns:
            print("\nHistorical data must have 'Dates' and 'Prices' columns for price analysis")
        else:
            print("\nGenerating historical price analysis...")
            model.plot_historical_price_analysis(df, save_plots=True)
    except Exception as e:
        print(f"\nHistorical data not available for price analysis: {e}")
    print("\nComplete analysis with visualizations finished!")
    print("Check for saved plot files: gas_storage_analysis.png, scenario_analysis.png")
    return results

if __name__ == "__main__":
    run_complete_analysis_with_plots()