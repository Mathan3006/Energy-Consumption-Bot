from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('electricity_model.pkl')
scaler = joblib.load('scaler.pkl')

# Track conversation state
conversation_state = {
    'inputs': [],
    'features': [
        'Global_reactive_power',
        'Voltage',
        'Global_intensity',
        'Sub_metering_1',
        'Sub_metering_2',
        'Sub_metering_3'
    ],
    'current': 0
}

@app.route('/')
def index():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    state = conversation_state

    try:
        val = float(user_input)
        state['inputs'].append(val)
        state['current'] += 1

        if state['current'] < len(state['features']):
            next_prompt = f"Enter {state['features'][state['current']]}:"
            return jsonify({'reply': next_prompt})
        else:
            input_array = np.array(state['inputs']).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]

            response = {
                'reply': f"✅ Predicted Global Active Power: {prediction:.2f} kW",
                'inputs': state['inputs'],
                'labels': state['features'],
                'prediction': prediction
            }

            state['inputs'] = []
            state['current'] = 0
            return jsonify(response)

    except ValueError:
        if state['current'] == 0:
            return jsonify({'reply': f"Welcome! Let's start. Enter {state['features'][state['current']]}:"})
        else:
            return jsonify({'reply': f"❌ Please enter a valid number for {state['features'][state['current']]}:"})

if __name__ == '__main__':
    app.run(debug=True)