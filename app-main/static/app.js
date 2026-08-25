const health = document.querySelector('#health');
const textInput = document.querySelector('#clinical-text');
const extractButton = document.querySelector('#extract-button');
const nlpResults = document.querySelector('#nlp-results');
const imageInput = document.querySelector('#image-input');
const imagePreview = document.querySelector('#image-preview');
const predictButton = document.querySelector('#predict-button');
const imageResult = document.querySelector('#image-result');

function showError(element, message) {
  element.className = 'results empty';
  element.textContent = message;
}

fetch('/api/health')
  .then(response => response.json())
  .then(data => {
    health.textContent = `${data.hpo_terms.toLocaleString()} HPO terms loaded`;
  })
  .catch(() => { health.textContent = 'Service unavailable'; });

extractButton.addEventListener('click', async () => {
  const text = textInput.value.trim();
  if (!text) { showError(nlpResults, 'Enter a clinical description first.'); return; }
  extractButton.disabled = true;
  nlpResults.textContent = 'Extracting...';
  try {
    const response = await fetch('/api/nlp/extract', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    nlpResults.className = 'results';
    nlpResults.innerHTML = data.results.length ? data.results.map(item =>
      `<div class="result"><span>${item.name}</span><small>${item.id} / ${item.score}</small></div>`
    ).join('') : '<span class="empty">No confident HPO terms found.</span>';
  } catch (error) { showError(nlpResults, error.message); }
  finally { extractButton.disabled = false; }
});

imageInput.addEventListener('change', () => {
  const file = imageInput.files[0];
  if (!file) { imagePreview.textContent = 'No image selected'; return; }
  const image = document.createElement('img');
  image.src = URL.createObjectURL(file);
  image.alt = 'Selected image preview';
  imagePreview.replaceChildren(image);
});

predictButton.addEventListener('click', async () => {
  const file = imageInput.files[0];
  if (!file) { showError(imageResult, 'Select an image first.'); return; }
  predictButton.disabled = true;
  imageResult.textContent = 'Classifying...';
  const formData = new FormData();
  formData.append('image', file);
  try {
    const response = await fetch('/api/image/predict', {method: 'POST', body: formData});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    imageResult.className = 'results';
    imageResult.innerHTML = `<div class="result"><span>${data.label}</span><small>${(data.confidence * 100).toFixed(1)}% confidence</small></div>`;
  } catch (error) { showError(imageResult, error.message); }
  finally { predictButton.disabled = false; }
});
