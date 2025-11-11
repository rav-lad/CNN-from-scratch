// DermaScan Frontend Application

class DermaScanApp {
    constructor() {
        this.selectedFile = null;
        this.apiBaseUrl = window.location.origin;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadConditions();
    }

    setupEventListeners() {
        // File input
        const fileInput = document.getElementById('fileInput');
        const selectFileBtn = document.getElementById('selectFileBtn');
        const uploadArea = document.getElementById('uploadArea');
        const changeImageBtn = document.getElementById('changeImageBtn');
        const analyzeBtn = document.getElementById('analyzeBtn');

        // Click to select file
        selectFileBtn.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('click', () => fileInput.click());

        // File selection
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('drag-over');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                this.selectedFile = file;
                this.showPreview(file);
            }
        });

        // Change image
        changeImageBtn.addEventListener('click', () => {
            fileInput.click();
        });

        // Analyze button
        analyzeBtn.addEventListener('click', () => this.analyzeImage());
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            // Validate file size (max 10MB)
            if (file.size > 10 * 1024 * 1024) {
                alert('Le fichier est trop volumineux. Taille maximale: 10MB');
                return;
            }

            // Validate file type
            if (!file.type.startsWith('image/')) {
                alert('Veuillez sélectionner une image (PNG, JPG, JPEG)');
                return;
            }

            this.selectedFile = file;
            this.showPreview(file);
        }
    }

    showPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const imagePreview = document.getElementById('imagePreview');
            imagePreview.src = e.target.result;

            // Hide upload area, show preview
            document.getElementById('uploadArea').style.display = 'none';
            document.getElementById('previewSection').style.display = 'block';
            document.getElementById('resultsSection').style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    async analyzeImage() {
        if (!this.selectedFile) {
            alert('Veuillez sélectionner une image');
            return;
        }

        const analyzeBtn = document.getElementById('analyzeBtn');
        const btnText = analyzeBtn.querySelector('.btn-text');
        const btnLoader = analyzeBtn.querySelector('.btn-loader');

        // Show loading state
        analyzeBtn.disabled = true;
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline-block';

        try {
            // Create form data
            const formData = new FormData();
            formData.append('file', this.selectedFile);

            // Send to API
            const response = await fetch(`${this.apiBaseUrl}/api/predict`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Erreur lors de l\'analyse');
            }

            const result = await response.json();
            this.displayResults(result);

        } catch (error) {
            console.error('Error:', error);
            alert('Une erreur est survenue lors de l\'analyse. Veuillez réessayer.');
        } finally {
            // Reset button state
            analyzeBtn.disabled = false;
            btnText.style.display = 'inline';
            btnLoader.style.display = 'none';
        }
    }

    displayResults(result) {
        const resultsSection = document.getElementById('resultsSection');
        const resultsContainer = document.getElementById('resultsContainer');

        if (!result.success || !result.predictions || result.predictions.length === 0) {
            resultsContainer.innerHTML = '<p>Aucun résultat trouvé.</p>';
            resultsSection.style.display = 'block';
            return;
        }

        // Clear previous results
        resultsContainer.innerHTML = '';

        // Display warning
        const warning = document.createElement('div');
        warning.className = 'warning-banner';
        warning.style.marginBottom = '2rem';
        warning.textContent = result.warning || 'Ceci n\'est pas un diagnostic médical.';
        resultsContainer.appendChild(warning);

        // Display each prediction
        result.predictions.forEach((prediction, index) => {
            const card = this.createResultCard(prediction, index);
            resultsContainer.appendChild(card);
        });

        // Show results section
        resultsSection.style.display = 'block';

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    createResultCard(prediction, index) {
        const card = document.createElement('div');

        // Determine severity class
        let severityClass = 'low-severity';
        let severityLabel = 'Faible';

        if (prediction.severity.toLowerCase().includes('high')) {
            severityClass = 'high-severity';
            severityLabel = 'Élevée';
        } else if (prediction.severity.toLowerCase().includes('moderate')) {
            severityClass = 'moderate-severity';
            severityLabel = 'Modérée';
        }

        card.className = `result-card ${severityClass}`;

        const confidencePercent = (prediction.confidence * 100).toFixed(1);

        card.innerHTML = `
            <div class="result-header">
                <h3 class="result-title">${index + 1}. ${prediction.condition}</h3>
                <span class="confidence-badge">${confidencePercent}%</span>
            </div>

            <div class="severity-badge severity-${severityClass.replace('-severity', '')}">
                Sévérité: ${severityLabel}
            </div>

            <p class="result-description">${prediction.description}</p>

            ${prediction.symptoms && prediction.symptoms.length > 0 ? `
                <div class="result-section">
                    <h3>Symptômes associés</h3>
                    <ul>
                        ${prediction.symptoms.map(s => `<li>${s}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}

            ${prediction.recommendations && prediction.recommendations.length > 0 ? `
                <div class="result-section">
                    <h3>Recommandations</h3>
                    <ul>
                        ${prediction.recommendations.map(r => `<li>${r}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}

            ${prediction.urgency ? `
                <div class="urgency-notice">
                    📅 ${prediction.urgency}
                </div>
            ` : ''}
        `;

        return card;
    }

    async loadConditions() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/conditions`);
            if (response.ok) {
                const data = await response.json();
                this.displayConditions(data.conditions);
            }
        } catch (error) {
            console.error('Error loading conditions:', error);
        }
    }

    displayConditions(conditions) {
        const conditionsList = document.getElementById('conditionsList');
        conditionsList.innerHTML = '';

        conditions.forEach(condition => {
            const tag = document.createElement('div');
            tag.className = 'condition-tag';
            tag.innerHTML = `<strong>${condition}</strong>`;
            conditionsList.appendChild(tag);
        });
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new DermaScanApp();
});
