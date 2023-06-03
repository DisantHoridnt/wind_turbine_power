from wind_turbine_power.model import data

def test_generate_turbine_data():
    wind_speeds, power_outputs = data.generate_turbine_data(100, 3.5, 15, 2, 25, random_state=42)
    
    assert len(wind_speeds) == 100
    assert len(power_outputs) == 100