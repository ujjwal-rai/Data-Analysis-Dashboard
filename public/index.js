document.getElementById('ask-button').addEventListener('click', async () => {
    const question = document.getElementById('question').value;
    const responseDiv = document.getElementById('chatbot-reply');
    responseDiv.innerHTML = 'Loading...'; // Show loading state

    try {
        const response = await fetch('/ask-chatbot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question })
        });

        const data = await response.json();
        if (response.ok) {
            responseDiv.innerHTML = `<strong>Chatbot:</strong> ${data.reply}`;
        } else {
            responseDiv.innerHTML = `<strong>Error:</strong> ${data.error}`;
        }
    } catch (error) {
        responseDiv.innerHTML = `<strong>Error:</strong> ${error.message}`;
    }
});