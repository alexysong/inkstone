# -*- coding: utf-8 -*-
import pandas as pd

def get_permittivity(file_name, wavelength):
    """
    Given a wavelength, as well as the csv file for the table of (n, k) against incident wavelength, 
    return the real part of permittivity.
    If there is no exact match, return the value for the closest wavelength.
    """
    # Load the CSV file
    data = pd.read_csv(file_name)

    # Convert columns to numeric values
    data['wl'] = pd.to_numeric(data['wl'], errors='coerce')
    data['n'] = pd.to_numeric(data['n'], errors='coerce')
    data['k'] = pd.to_numeric(data['k'], errors='coerce')
    # Find the closest wavelength in the dataset
    closest_wavelength = data.iloc[(data['wl'] - wavelength).abs().argmin()]['wl']
    n = data.loc[data['wl'] == closest_wavelength, 'n'].values[0]
    k = data.loc[data['wl'] == closest_wavelength, 'k'].values[0]
    epsilon_r = n**2 - k**2  # Real part of permittivity
    epsilon_i = 2 * n * k    # Imaginary part of permittivity
    return epsilon_r

def save_simulation_results(file_path='data.xlsx', **kwargs):
    """
    Saves simulation results with flexible columns to an Excel file.

    Parameters:
        file_path (str): Path to save the Excel file (default: 'data.xlsx').
        **kwargs: Arbitrary keyword arguments where the key is the column name 
                  and the value is the data (list or array) for that column.

    Returns:
        None
    """
    # Create a DataFrame from the provided keyword arguments
    data = {key: value for key, value in kwargs.items()}
    df = pd.DataFrame(data)

    # Export the DataFrame to an Excel file
    df.to_excel(file_path, index=False)

    print(f"Data successfully exported to {file_path}")

"""
# Example usage: Save Reflection, Transmission, and other custom data
save_simulation_results(
    file_path='custom_results.xlsx',
    Reflection=[0.4, 0.5, 0.6],
    Transmission=[0.6, 0.5, 0.4],
    CustomData=[10, 20, 30]
)
"""


def get_refractive_index(file_name, wavelength):
    """
    Given a wavelength, as well as the csv file for the table of refractive index against incident wavelength, return the refractive index.
    If there is no exact match, return the value for the closest wavelength.
    """
    # Load the CSV file
    data = pd.read_csv(file_name)

    # Convert columns to numeric values
    data['wl'] = pd.to_numeric(data['wl'], errors='coerce')
    data['n'] = pd.to_numeric(data['n'], errors='coerce')
    # Find the closest wavelength in the dataset
    closest_wavelength = data.iloc[(data['wl'] - wavelength).abs().argmin()]['wl']
    refractive_index = data.loc[data['wl'] == closest_wavelength, 'n'].values[0]
    return refractive_index
