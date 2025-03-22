function copyToClipboard() {
    // Get the text to copy
    let textToCopy = document.getElementById("coverlettercontent").innerText;

    // Copy to clipboard
    navigator.clipboard.writeText(textToCopy).then(() => {
        let button = document.getElementById("copyButton");
        let originalText = button.innerText;

        // Change button text to "Copied"
        button.innerText = "Copied!";
        button.disabled = true; // Optionally disable button for 1 second

        setTimeout(() => {
            button.innerText = originalText;
            button.disabled = false;
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy:', err);
    });
}

document.getElementById("copyButton").addEventListener("click", () => {
    copyToClipboard();
});

