$(document).ready(function() {
    // Initialize Select2 for symptom selection
    $('.symptom-select').select2({
        placeholder: "Search and select symptoms...",
        maximumSelectionLength: 10,
        width: '100%',
        theme: 'classic'
    });

    // Handle form validation
    $('#prediction-form').on('submit', function(e) {
        const symptoms = $('.symptom-select').val();
        if (!symptoms || symptoms.length < 3) {
            e.preventDefault();
            alert("Please select at least 3 symptoms to predict.");
        }
    });

    // Dynamic tag display (optional enhancement)
    $('.symptom-select').on('change', function() {
        const selected = $(this).val();
        const container = $('#selected-tags');
        container.empty();
        if (selected) {
            selected.forEach(s => {
                container.append(`<span class="tag">${s}</span>`);
            });
        }
    });
});
