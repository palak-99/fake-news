document.addEventListener('DOMContentLoaded', function () {
    const newsForm = document.getElementById('newsForm');
    const newsInput = document.getElementById('newsInput');
    const detectBtn = document.getElementById('detectBtn');
    const resultDiv = document.getElementById('result');

    newsForm.addEventListener('submit', function (event) {
        event.preventDefault();

        // Get the news article from the textarea
        const newsArticle = newsInput.value.trim();

        // Simulate fake news detection (replace this with your actual detection logic)
        const isFake = Math.random() < 0.5; // Randomly determine if the news is fake or real

        // Display the result with animations
        resultDiv.innerHTML = '';
        const message = isFake ? "This news seems to be fake." : "This news appears to be genuine.";
        const resultColor = isFake ? "#ff0000" : "#008000 ";

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
    });
});
