from Filter.Utils import load_data

df = load_data('data/')

df.to_pickle('FilterDF.pkl')
