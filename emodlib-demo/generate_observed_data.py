"""
Generate synthetic observed prevalence-by-age data for calibration.

This script creates a CSV file with example prevalence-by-age data that
serves as the calibration target. In a real scenario, this would be replaced
with actual field survey data.

The example data pattern:
- Low prevalence in infants (maternal immunity)
- Rising prevalence in young children (exposure builds)
- Peak prevalence around age 3-5 (high exposure, limited immunity)
- Gradual decline in older children/adults (acquired immunity)
- Plateau in adults (equilibrium between exposure and immunity)
"""

import polars as pl


def generate_synthetic_prevalence() -> pl.DataFrame:
    """
    Generate synthetic prevalence-by-age data.

    This creates an example dataset with realistic malaria prevalence patterns
    based on typical endemic area surveys. The pattern shows:
    - Age-dependent exposure (via biting risk)
    - Acquired immunity effects (declining prevalence with age)

    Returns:
        DataFrame with columns: age_years, prevalence
    """
    # Example prevalence pattern from optimize.py
    # This represents a high-transmission seasonal setting
    age_years = list(range(20))
    prevalence = [
        0.2, 0.4, 0.7, 0.9, 0.85,
        0.85, 0.8, 0.8, 0.75, 0.75,
        0.7, 0.65, 0.6, 0.5, 0.45,
        0.4, 0.35, 0.3, 0.25, 0.25,
    ]

    return pl.DataFrame({
        'age_years': age_years,
        'prevalence': prevalence,
    })


def main():
    """Generate and save observed prevalence data."""
    df = generate_synthetic_prevalence()

    # Save to CSV
    output_path = "data/observed_prevalence.csv"
    df.write_csv(output_path)

    print(f"Generated synthetic observed data: {output_path}")
    print(f"\nData summary:")
    print(df)
    print(f"\nMean prevalence: {df['prevalence'].mean():.3f}")
    print(f"Peak prevalence: {df['prevalence'].max():.3f} (age {df.filter(pl.col('prevalence') == pl.col('prevalence').max())['age_years'][0]})")


if __name__ == '__main__':
    main()
