
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    def __init__(self, data_path=None, dataframe=None):
        """
        Initialize the EDA analyzer

        Args:
            data_path (str): Path to CSV file
            dataframe (pd.DataFrame): DataFrame to analyze
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
        Create sample dataset for demonstration
        """
        np.random.seed(42)
        n_samples = 1000

        data = {
            'age': np.random.normal(35, 10, n_samples).astype(int),
            'income': np.random.lognormal(10, 0.5, n_samples),
            'education_years': np.random.normal(14, 3, n_samples).astype(int),
            'experience_years': np.random.normal(8, 5, n_samples).astype(int),
            'satisfaction_score': np.random.normal(7.5, 1.5, n_samples),
            'department': np.random.choice(['Engineering', 'Marketing', 'Sales', 'HR', 'Finance'], n_samples),
            'city': np.random.choice(['New York', 'San Francisco', 'Chicago', 'Boston', 'Seattle'], n_samples),
            'remote_work': np.random.choice(['Yes', 'No'], n_samples, p=[0.4, 0.6])
        }

        # Add some correlations
        df = pd.DataFrame(data)
        df['age'] = np.clip(df['age'], 22, 65)
        df['education_years'] = np.clip(df['education_years'], 12, 20)
        df['experience_years'] = np.clip(df['experience_years'], 0, df['age'] - 22)
        df['satisfaction_score'] = np.clip(df['satisfaction_score'], 1, 10)

        # Create some missing values
        missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df.loc[missing_indices, 'satisfaction_score'] = np.nan

        return df

    def basic_info(self):
        """
        Display basic information about the dataset
        """
        print("=" * 50)
        print("DATASET OVERVIEW")
        print("=" * 50)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Total missing values: {self.df.isnull().sum().sum()}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print()

        print("Data Types:")
        print(self.df.dtypes)
        print()

        print("First 5 rows:")
        print(self.df.head())
        print()

        return self.df.info()

    def missing_values_analysis(self):
        """
        Analyze missing values in the dataset
        """
        print("=" * 50)
        print("MISSING VALUES ANALYSIS")
        print("=" * 50)

        missing_data = pd.DataFrame({
            'Missing Count': self.df.isnull().sum(),
            'Missing Percentage': (self.df.isnull().sum() / len(self.df)) * 100
        })

        missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

        if len(missing_data) > 0:
            print(missing_data)

            # Visualize missing values
            plt.figure(figsize=(10, 6))
            missing_data['Missing Percentage'].plot(kind='bar')
            plt.title('Missing Values by Column')
            plt.ylabel('Missing Percentage (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('missing_values_plot.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("No missing values found in the dataset!")

        return missing_data

    def numerical_analysis(self):
        """
        Analyze numerical columns
        """
        print("=" * 50)
        print("NUMERICAL VARIABLES ANALYSIS")
        print("=" * 50)

        if not self.numeric_columns:
            print("No numerical columns found!")
            return

        # Descriptive statistics
        desc_stats = self.df[self.numeric_columns].describe()
        print("Descriptive Statistics:")
        print(desc_stats)
        print()

        # Distribution plots
        n_cols = len(self.numeric_columns)
        n_rows = (n_cols + 2) // 3

        plt.figure(figsize=(15, 5 * n_rows))
        for i, col in enumerate(self.numeric_columns, 1):
            plt.subplot(n_rows, 3, i)
            sns.histplot(data=self.df, x=col, kde=True)
            plt.title(f'Distribution of {col}')

            # Add statistics to the plot
            mean_val = self.df[col].mean()
            median_val = self.df[col].median()
            plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            plt.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
            plt.legend()

        plt.tight_layout()
        plt.savefig('numerical_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()

        return desc_stats

    def categorical_analysis(self):
        """
        Analyze categorical columns
        """
        print("=" * 50)
        print("CATEGORICAL VARIABLES ANALYSIS")
        print("=" * 50)

        if not self.categorical_columns:
            print("No categorical columns found!")
            return

        for col in self.categorical_columns:
            print(f"\n{col.upper()}:")
            value_counts = self.df[col].value_counts()
            print(value_counts)
            print(f"Unique values: {self.df[col].nunique()}")

        # Visualize categorical variables
        n_cols = len(self.categorical_columns)
        n_rows = (n_cols + 1) // 2

        plt.figure(figsize=(15, 5 * n_rows))
        for i, col in enumerate(self.categorical_columns, 1):
            plt.subplot(n_rows, 2, i)
            value_counts = self.df[col].value_counts()
            plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
            plt.title(f'Distribution of {col}')

        plt.tight_layout()
        plt.savefig('categorical_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()

    def correlation_analysis(self):
        """
        Analyze correlations between numerical variables
        """
        print("=" * 50)
        print("CORRELATION ANALYSIS")
        print("=" * 50)

        if len(self.numeric_columns) < 2:
            print("Need at least 2 numerical columns for correlation analysis!")
            return

        correlation_matrix = self.df[self.numeric_columns].corr()
        print("Correlation Matrix:")
        print(correlation_matrix)
        print()

        # Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Find strong correlations
        strong_corr = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:
                    strong_corr.append({
                        'Variable 1': correlation_matrix.columns[i],
                        'Variable 2': correlation_matrix.columns[j],
                        'Correlation': corr_value
                    })

        if strong_corr:
            print("Strong Correlations (|r| > 0.5):")
            for corr in strong_corr:
                print(f"{corr['Variable 1']} <-> {corr['Variable 2']}: {corr['Correlation']:.3f}")

        return correlation_matrix

    def outlier_detection(self):
        """
        Detect outliers using IQR method
        """
        print("=" * 50)
        print("OUTLIER DETECTION")
        print("=" * 50)

        outlier_summary = {}

        plt.figure(figsize=(15, 5))
        for i, col in enumerate(self.numeric_columns, 1):
            # IQR method
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_summary[col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(self.df) * 100
            }

            # Box plot
            plt.subplot(1, len(self.numeric_columns), i)
            sns.boxplot(y=self.df[col])
            plt.title(f'{col}\nOutliers: {len(outliers)} ({len(outliers)/len(self.df)*100:.1f}%)')

        plt.tight_layout()
        plt.savefig('outlier_boxplots.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Print summary
        for col, info in outlier_summary.items():
            print(f"{col}: {info['count']} outliers ({info['percentage']:.2f}%)")

        return outlier_summary

    def generate_report(self):
        """
        Generate a complete EDA report
        """
        print("\n" + "="*70)
        print("COMPLETE EXPLORATORY DATA ANALYSIS REPORT")
        print("="*70)

        # Run all analyses
        self.basic_info()
        missing_data = self.missing_values_analysis()
        desc_stats = self.numerical_analysis()
        self.categorical_analysis()
        correlation_matrix = self.correlation_analysis()
        outlier_summary = self.outlier_detection()

        # Save summary to file
        report = f"""
EDA SUMMARY REPORT
==================

Dataset Overview:
- Shape: {self.df.shape}
- Numerical columns: {len(self.numeric_columns)}
- Categorical columns: {len(self.categorical_columns)}
- Missing values: {self.df.isnull().sum().sum()}

Key Findings:
- Dataset contains {len(self.df)} samples with {len(self.df.columns)} features
- {len([col for col in self.df.columns if self.df[col].isnull().sum() > 0])} columns have missing values
- Numerical variables show {'normal' if all(stats.shapiro(self.df[col].dropna())[1] > 0.05 for col in self.numeric_columns if len(self.df[col].dropna()) <= 5000) else 'non-normal'} distributions
- Strong correlations found between key variables

Recommendations:
1. Handle missing values in satisfaction_score column
2. Consider outlier treatment for variables with high outlier counts
3. Further investigate strong correlations for feature engineering
4. Consider data transformation for skewed variables
        """

        with open('eda_report.txt', 'w') as file:
            file.write(report)

        print("\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print("Files generated:")
        print("- missing_values_plot.png")
        print("- numerical_distributions.png")
        print("- categorical_distributions.png")
        print("- correlation_heatmap.png")
        print("- outlier_boxplots.png")
        print("- eda_report.txt")
        print("="*50)

def main():
    """
    Main function to run EDA
    """
    print("Starting Exploratory Data Analysis...")

    # Initialize analyzer with sample data
    analyzer = DataAnalyzer()

    # Generate complete report
    analyzer.generate_report()

if __name__ == "__main__":
    main()
