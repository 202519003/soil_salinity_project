with open('data/raw/weather/Anand_weather.csv', 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines[:35]):
        print(f'{i}: {line.strip()}')