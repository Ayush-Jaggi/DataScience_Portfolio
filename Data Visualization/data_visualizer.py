
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataVisualizer:
    def __init__(self, data_path=None, dataframe=None):
        """
        Initialize the Data Visualizer

        Args:
            data_path (str): Path to CSV file
            dataframe (pd.DataFrame): DataFrame to visualize
        """
        if data_path:
            self.df = pd.read_csv(data_path)
        elif dataframe is not None:
            self.df = dataframe.copy()
        else:
            # Create sample dataset for demonstration
            self.df = self.create_sample_data()

        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()

    def create_sample_data(self):
        """
        Create sample dataset for visualization demo
        """
        np.random.seed(42)
        n_samples = 500

        # Create realistic sales data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
        regions = ['North', 'South', 'East', 'West', 'Central']

        data = []
        for month in months:
            for product in products:
                for region in regions:
                    # Create realistic sales patterns
                    base_sales = np.random.normal(1000, 200)
                    seasonal_factor = 1.2 if month in ['Nov', 'Dec'] else 1.0
                    product_factor = {'Product A': 1.5, 'Product B': 1.2, 'Product C': 1.0, 
                                    'Product D': 0.8, 'Product E': 0.9}[product]

                    sales = base_sales * seasonal_factor * product_factor
                    revenue = sales * np.random.uniform(50, 150)  # Price per unit

                    data.append({
                        'month': month,
                        'product': product,
                        'region': region,
                        'sales_units': max(int(sales), 0),
                        'revenue': max(revenue, 0),
                        'profit_margin': np.random.uniform(0.1, 0.3),
                        'customer_satisfaction': np.random.normal(4.0, 0.5),
                        'marketing_spend': np.random.uniform(5000, 15000)
                    })

        df = pd.DataFrame(data)
        df['profit'] = df['revenue'] * df['profit_margin']
        df['customer_satisfaction'] = np.clip(df['customer_satisfaction'], 1, 5)

        return df

    def create_basic_plots(self):
        """
        Create basic visualization plots
        """
        print("Creating basic visualization plots...")

        # Set up the plotting area
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Basic Data Visualizations', fontsize=16, fontweight='bold')

        # 1. Bar Plot - Sales by Product
        product_sales = self.df.groupby('product')['sales_units'].sum().sort_values(ascending=False)
        axes[0, 0].bar(product_sales.index, product_sales.values, color='skyblue', alpha=0.8)
        axes[0, 0].set_title('Total Sales by Product', fontweight='bold')
        axes[0, 0].set_xlabel('Product')
        axes[0, 0].set_ylabel('Sales Units')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. Line Plot - Revenue Trend by Month
        monthly_revenue = self.df.groupby('month')['revenue'].sum()
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_revenue = monthly_revenue.reindex(month_order)

        axes[0, 1].plot(monthly_revenue.index, monthly_revenue.values, 
                       marker='o', linewidth=2, markersize=6, color='green')
        axes[0, 1].set_title('Monthly Revenue Trend', fontweight='bold')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Revenue ($)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Scatter Plot - Revenue vs Marketing Spend
        axes[1, 0].scatter(self.df['marketing_spend'], self.df['revenue'], 
                          alpha=0.6, color='purple')
        axes[1, 0].set_title('Revenue vs Marketing Spend', fontweight='bold')
        axes[1, 0].set_xlabel('Marketing Spend ($)')
        axes[1, 0].set_ylabel('Revenue ($)')

        # Add trend line
        z = np.polyfit(self.df['marketing_spend'], self.df['revenue'], 1)
        p = np.poly1d(z)
        axes[1, 0].plot(self.df['marketing_spend'], p(self.df['marketing_spend']), 
                       "r--", alpha=0.8, linewidth=2)

        # 4. Histogram - Customer Satisfaction Distribution
        axes[1, 1].hist(self.df['customer_satisfaction'], bins=20, 
                       color='orange', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Customer Satisfaction Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Satisfaction Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(self.df['customer_satisfaction'].mean(), 
                          color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {self.df["customer_satisfaction"].mean():.2f}')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig('basic_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_advanced_plots(self):
        """
        Create advanced visualization plots
        """
        print("Creating advanced visualization plots...")

        # Set up the plotting area
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Advanced Data Visualizations', fontsize=16, fontweight='bold')

        # 1. Heatmap - Sales by Product and Region
        pivot_data = self.df.pivot_table(values='sales_units', 
                                        index='product', 
                                        columns='region', 
                                        aggfunc='mean')
        sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', 
                   ax=axes[0, 0], cbar_kws={'label': 'Average Sales Units'})
        axes[0, 0].set_title('Average Sales Heatmap by Product and Region', fontweight='bold')

        # 2. Box Plot - Revenue Distribution by Region
        sns.boxplot(data=self.df, x='region', y='revenue', ax=axes[0, 1])
        axes[0, 1].set_title('Revenue Distribution by Region', fontweight='bold')
        axes[0, 1].set_xlabel('Region')
        axes[0, 1].set_ylabel('Revenue ($)')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. Violin Plot - Profit Margin by Product
        sns.violinplot(data=self.df, x='product', y='profit_margin', ax=axes[1, 0])
        axes[1, 0].set_title('Profit Margin Distribution by Product', fontweight='bold')
        axes[1, 0].set_xlabel('Product')
        axes[1, 0].set_ylabel('Profit Margin')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. Correlation Heatmap
        numeric_df = self.df[['sales_units', 'revenue', 'profit_margin', 
                             'customer_satisfaction', 'marketing_spend', 'profit']]
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', ax=axes[1, 1])
        axes[1, 1].set_title('Correlation Matrix', fontweight='bold')

        plt.tight_layout()
        plt.savefig('advanced_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_interactive_plots(self):
        """
        Create interactive plots using Plotly
        """
        print("Creating interactive visualizations...")

        # 1. Interactive Scatter Plot
        fig1 = px.scatter(self.df, x='marketing_spend', y='revenue', 
                         color='region', size='sales_units',
                         hover_data=['product', 'profit_margin'],
                         title='Interactive: Revenue vs Marketing Spend by Region')
        fig1.write_html('interactive_scatter.html')

        # 2. Interactive Line Chart
        monthly_data = self.df.groupby(['month', 'product'])['revenue'].sum().reset_index()
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_data['month'] = pd.Categorical(monthly_data['month'], categories=month_order, ordered=True)
        monthly_data = monthly_data.sort_values('month')

        fig2 = px.line(monthly_data, x='month', y='revenue', color='product',
                      title='Interactive: Monthly Revenue Trend by Product')
        fig2.write_html('interactive_line.html')

        # 3. Interactive Bar Chart
        region_summary = self.df.groupby('region').agg({
            'sales_units': 'sum',
            'revenue': 'sum',
            'profit': 'sum'
        }).reset_index()

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name='Sales Units', x=region_summary['region'], 
                             y=region_summary['sales_units'], yaxis='y'))
        fig3.add_trace(go.Bar(name='Revenue', x=region_summary['region'], 
                             y=region_summary['revenue'], yaxis='y2'))

        fig3.update_layout(
            title='Interactive: Sales and Revenue by Region',
            xaxis_title='Region',
            yaxis=dict(title='Sales Units', side='left'),
            yaxis2=dict(title='Revenue ($)', side='right', overlaying='y'),
            barmode='group'
        )
        fig3.write_html('interactive_bar.html')

        # 4. Interactive Sunburst Chart
        fig4 = px.sunburst(self.df, path=['region', 'product'], values='revenue',
                          title='Interactive: Revenue Distribution by Region and Product')
        fig4.write_html('interactive_sunburst.html')

        print("Interactive plots saved as HTML files:")
        print("- interactive_scatter.html")
        print("- interactive_line.html")
        print("- interactive_bar.html")
        print("- interactive_sunburst.html")

    def create_dashboard_plots(self):
        """
        Create a comprehensive dashboard
        """
        print("Creating dashboard visualizations...")

        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Sales by Product', 'Monthly Revenue Trend', 
                           'Revenue vs Marketing Spend', 'Regional Performance',
                           'Customer Satisfaction', 'Profit Analysis'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # 1. Sales by Product
        product_sales = self.df.groupby('product')['sales_units'].sum()
        fig.add_trace(go.Bar(x=product_sales.index, y=product_sales.values, 
                            name='Sales', marker_color='skyblue'), row=1, col=1)

        # 2. Monthly Revenue Trend
        monthly_revenue = self.df.groupby('month')['revenue'].sum()
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_revenue = monthly_revenue.reindex(month_order)
        fig.add_trace(go.Scatter(x=monthly_revenue.index, y=monthly_revenue.values,
                                mode='lines+markers', name='Revenue', 
                                line=dict(color='green', width=3)), row=1, col=2)

        # 3. Revenue vs Marketing Spend
        fig.add_trace(go.Scatter(x=self.df['marketing_spend'], y=self.df['revenue'],
                                mode='markers', name='Revenue vs Marketing',
                                marker=dict(color='purple', opacity=0.6)), row=2, col=1)

        # 4. Regional Performance
        region_revenue = self.df.groupby('region')['revenue'].sum()
        fig.add_trace(go.Bar(x=region_revenue.index, y=region_revenue.values,
                            name='Regional Revenue', marker_color='orange'), row=2, col=2)

        # 5. Customer Satisfaction
        fig.add_trace(go.Histogram(x=self.df['customer_satisfaction'], 
                                  name='Satisfaction', marker_color='red'), row=3, col=1)

        # 6. Profit Analysis
        product_profit = self.df.groupby('product')['profit'].sum()
        fig.add_trace(go.Bar(x=product_profit.index, y=product_profit.values,
                            name='Profit', marker_color='darkgreen'), row=3, col=2)

        # Update layout
        fig.update_layout(height=900, showlegend=False, 
                         title_text="Business Performance Dashboard")
        fig.write_html('dashboard.html')

        print("Dashboard saved as dashboard.html")

    def save_sample_data(self):
        """
        Save sample data to CSV
        """
        self.df.to_csv('sample_business_data.csv', index=False)
        print("Sample data saved to sample_business_data.csv")

    def generate_all_visualizations(self):
        """
        Generate all visualization types
        """
        print("=" * 60)
        print("GENERATING COMPLETE VISUALIZATION SUITE")
        print("=" * 60)

        # Save sample data
        self.save_sample_data()

        # Create all visualizations
        self.create_basic_plots()
        self.create_advanced_plots()
        self.create_interactive_plots()
        self.create_dashboard_plots()

        print("\n" + "=" * 60)
        print("ALL VISUALIZATIONS GENERATED!")
        print("Files created:")
        print("- basic_plots.png")
        print("- advanced_plots.png")
        print("- interactive_scatter.html")
        print("- interactive_line.html")
        print("- interactive_bar.html")
        print("- interactive_sunburst.html")
        print("- dashboard.html")
        print("- sample_business_data.csv")
        print("=" * 60)

def main():
    """
    Main function to run data visualization
    """
    print("Starting Data Visualization Suite...")

    # Initialize visualizer
    visualizer = DataVisualizer()

    # Generate all visualizations
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main()
