import express from 'express';
import multer from 'multer';
import { exec } from 'child_process';
import fetch from 'node-fetch';
import dotenv from 'dotenv';
import path from 'path';

// Load environment variables
dotenv.config();

const app = express();

// Set up static files to serve HTML
app.use(express.static('public'));
app.use(express.json()); // Middleware to parse JSON bodies

// Set up multer for file uploads
const upload = multer({ dest: 'uploads/' });

// Route to handle file upload
app.post('/upload', upload.fields([{ name: 'dataset' }, { name: 'features' }]), (req, res) => {
    const datasetPath = req.files['dataset'][0].path;
    const featuresPath = req.files['features'][0].path;

    // Execute the Python script for logistic regression
    exec(`python logistic_regression.py ${datasetPath} ${featuresPath}`, (error, stdout, stderr) => {
        if (error) {
            return res.status(500).send(`Error: ${stderr}`);
        }
        res.send(`<h3>Logistic Regression Results:</h3><pre>${stdout}</pre>`);
    });
});

// Route to ask the chatbot
app.post('/ask-chatbot', async (req, res) => {
    const question = req.body.question;

    try {
        const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: "llama3-8b-8192", // Use the correct model
                messages: [
                    { role: "user", content: question } // Format user input correctly
                ],
                max_tokens: 150 // Set max tokens as needed
            })
        });

        if (!response.ok) {
            const errorResponse = await response.json();
            console.error('API Error:', errorResponse); // Log the error for debugging
            return res.status(400).json({ error: errorResponse.error.message });
        }

        const data = await response.json();
        console.log('Full API response:', JSON.stringify(data, null, 2)); // Log to inspect the entire response

        // Check if the reply exists in the response
        if (data && data.choices && data.choices.length > 0) {
            res.json({ reply: data.choices[0].message.content }); // Correctly extract the reply
        } else {
            res.status(500).json({ error: "Unexpected response format from chatbot API" });
        }
    } catch (error) {
        console.error("Error contacting chatbot API:", error);
        res.status(500).json({ error: "Error contacting chatbot API" });
    }
});

// Start server
const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
