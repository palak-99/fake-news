document.addEventListener('DOMContentLoaded', function () {
    const newsForm = document.getElementById('newsForm');
    const newsInput = document.getElementById('newsInput');
    const detectBtn = document.getElementById('detectBtn');
    const resultDiv = document.getElementById('result');

    newsForm.addEventListener('submit', async function (event) {
        event.preventDefault();

        // Get the news article from the textarea
        const newsArticle = newsInput.value.trim();

        const url = `http://127.0.0.1:5000/news/${encodeURIComponent(newsArticle)}`;

        // Define fetch options
        const options = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
        };

        try {
            // Make the POST request
            const response = await fetch(url, options);
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            const apiResponse = data.res;

            const isFake = apiResponse === 'No, It is not fake';

            // Display the result with animations
            resultDiv.innerHTML = '';
            const message = apiResponse;
            const resultColor = isFake ? "#008000" : "#ff0000";

            const messageElement = document.createElement('p');
            messageElement.textContent = message;
            messageElement.style.color = resultColor;
            resultDiv.appendChild(messageElement);

            // Add animation
            messageElement.animate([
                { transform: 'scale(0.5)', opacity: 0 },
                { transform: 'scale(1)', opacity: 1 }
            ], {
                duration: 500,
                easing: 'ease-in-out'
            });
        } catch (error) {
            console.error('There was a problem with your fetch operation:', error);
        }
    });
});
