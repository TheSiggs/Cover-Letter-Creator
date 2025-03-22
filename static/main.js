function scrollToGeneratedText() {
    setTimeout(() => {
        document.getElementById('coverletter').scrollIntoView({ behavior: 'smooth' });
    }, 200);
}

document.addEventListener("htmx:beforeRequest", function(event) {
    var modal = new bootstrap.Modal(document.getElementById('loadingModal'));
    modal.show();
});

document.addEventListener("htmx:afterRequest", function(event) {
    setTimeout(() => {
        document.getElementById('coverletter').classList.remove('hidden');
        var modalEl = document.getElementById('loadingModal');
        var modal = bootstrap.Modal.getInstance(modalEl);
        if (modal) {
            modal.hide();
        }
        scrollToGeneratedText();
    }, 2000);
});

