import json

with open('cable_params.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

spectrum_keys = [k for k in data.keys() if 'spectrum' in k]
print(f'Spectrum entries: {len(spectrum_keys)}')
for key in spectrum_keys[:3]:
    spec = data[key]
    print(f'  {key}: {len(spec.get("freq_mhz", []))} frequency points')
    print(f'    Range: {spec["freq_mhz"][0]:.2f} - {spec["freq_mhz"][-1]:.2f} MHz')
