function processQuery() {
    const query = document.getElementById('userInputQuery').value;
    spinner = document.getElementById('searchSpinner')
    spinner.classList.remove('visually-hidden');

    const versionOption1 = document.getElementById("versionOption1");
    const version = versionOption1.checked ? "full" : "lite";
    
    fetch('/query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            text: query,
            version: version,
        }),
    })
    .then(response => response.json())
    .then(data => {

        const omniResult = data.omniquery;
        const ragResult = data.rag;

        console.log(omniResult);
        console.log(omniResult.images)

        document.getElementById('displayAnswerOmniQuery').innerText = omniResult.answer;
        document.getElementById('displayExplanationOmniQuery').innerText = omniResult.explanation;
        displayImages(omniResult.images, 'imageContainerOmniQuery');

        // document.getElementById('displayAnswerRAG').innerText = ragResult.answer;
        // document.getElementById('displayExplanationRAG').innerText = ragResult.explanation;
        // displayImages(ragResult.images, 'imageContainerRAG');
        spinner.classList.add('visually-hidden');
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

function displayImages(images, divID) {
    const container = document.getElementById(divID);
    container.innerHTML = '';  // Clear existing images
    container.style.position = 'relative';
    images.forEach(imageBase64 => {
        const img = document.createElement('img');
        const raw_data = imageBase64['raw_data'];
        //  imageBase64 is a dict/json
        // img.src = `data:image/jpeg;base64,${imageBase64}`;
        img.src = `data:image/jpeg;base64,${imageBase64['image']}`;
        img.alt = "Retrieved image";
        // img.style.height = '400px';  // Adjust the size as needed
        img.style.maxHeight = '300px';
        img.style.maxWidth = '300px';
        img.style.cursor = 'pointer';
        // img.style.position = 'relative';

        // Create a tooltip div to show metadata
        const address = raw_data['metadata']['location']?.['address'] || 'No address available';

        const tooltip = document.createElement('div');
        tooltip.style.position = 'absolute';
        tooltip.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
        tooltip.style.color = '#fff';
        tooltip.style.padding = '5px';
        tooltip.style.borderRadius = '5px';
        tooltip.style.visibility = 'hidden';
        tooltip.style.zIndex = '10';
        tooltip.style.maxWidth = '300px';
        tooltip.style.fontSize = '12px';
        tooltip.innerText = `Filename: ${raw_data['filename']}\nDate: ${raw_data['metadata']['temporal_info']['date_string']}\nLocation: ${address}`;  // Display metadata
        
        // Show tooltip on hover
        img.addEventListener('mouseover', function() {
            tooltip.style.visibility = 'visible';
        });
        
        // Hide tooltip when not hovering
        img.addEventListener('mouseout', function() {
            tooltip.style.visibility = 'hidden';
        });
        
        // Position the tooltip near the cursor
        img.addEventListener('mousemove', function(event) {
            tooltip.style.top = `${event.offsetY + 10}px`;  // Adjust position as needed
            tooltip.style.left = `${event.offsetX + 10}px`;
        });

        container.appendChild(img);
        container.appendChild(tooltip);
        container.appendChild(img);
    });
}

function init() {
    const initButton = document.getElementById('initButton');
    const apiKey = "";
    const folderPath = "";
    
    fetch('/init', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            'api_key': apiKey,
            'folder_path': folderPath
        }) 
    })
    .then(response => response.json())
    .then(data => {
        console.log('Response:', data);
        initButton.disabled = true;
        initButton.innerText = 'Initialized';
        initButton.classList.remove('btn-outline-dark');
        initButton.classList.add('btn-secondary');

    })
    .catch((error) => {
        console.error('Error:', error);
    });
}


function saveResult() {

    const query = document.getElementById('userInputQuery').value;
    const omniAnswer = document.getElementById('displayAnswerOmniQuery').innerText;
    const omniExplanation = document.getElementById('displayExplanationOmniQuery').innerText;
    const ragAnswer = document.getElementById('displayAnswerRAG').innerText;
    const ragExplanation = document.getElementById('displayExplanationRAG').innerText;


    const selections = {};
    const allRadioButtons = document.querySelectorAll('#ratingContainerOmniQuery input[type="radio"], #ratingContainerRAG input[type="radio"]');

    allRadioButtons.forEach(radio => {
      if (radio.checked) {
        const [criterion, containerId] = radio.name.split('-').slice(1); 

        const label = document.querySelector(`label[for="${radio.id}"]`);
        const value = label ? label.textContent : 'N/A';
        selections[criterion + '-' + containerId] = value;
      }
      radio.checked = false;
    });

    selections['query'] = query;
    selections['omniAnswer'] = omniAnswer;
    selections['omniExplanation'] = omniExplanation;
    selections['ragAnswer'] = ragAnswer;
    selections['ragExplanation'] = ragExplanation;

    console.log(selections);

    fetch('/save_result', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(selections)
    })
    .then(response => response.json())
    .then(data => {
        console.log('Response:', data);

        // Clear the query input and answer
        document.getElementById('userInputQuery').value = '';
        document.getElementById('displayAnswerOmniQuery').innerText = '';
        document.getElementById('displayExplanationOmniQuery').innerText = '';
        document.getElementById('imageContainerOmniQuery').innerHTML = '';

        document.getElementById('displayAnswerRAG').innerText = '';
        document.getElementById('displayExplanationRAG').innerText = '';
        document.getElementById('imageContainerRAG').innerHTML = '';

    })
    .catch((error) => {
        console.error('Error:', error);
    });

}

function createRatingSection(criteria, containerId) {
    const container = document.createElement('div');
    container.className = 'd-flex flex-row mb-3 justify-content-center';
    container.style.marginTop = '10px';

    const label = document.createElement('div');
    label.className = 'p-2';
    label.style.width = '110px';
    label.style.textAlign = 'right';
    label.textContent = `${criteria}:`;

    const btnGroupDiv = document.createElement('div');
    btnGroupDiv.className = 'col-auto';

    const btnGroup = document.createElement('div');
    btnGroup.className = 'btn-group';
    btnGroup.setAttribute('role', 'group');
    btnGroup.setAttribute('aria-label', `Basic ${criteria.toLowerCase()} rating group`);

    for (let i = 1; i <= 5; i++) {
      const input = document.createElement('input');
      input.type = 'radio';
      input.className = 'btn-check';
      input.name = `radio-${criteria.toLowerCase()}-${containerId}`;
      input.id = `${criteria.toLowerCase()}${i}-${containerId}`;
      input.autocomplete = 'off';

      const label = document.createElement('label');
      label.className = 'btn btn-outline-secondary';
      label.setAttribute('for', input.id);
      label.textContent = i;

      btnGroup.appendChild(input);
      btnGroup.appendChild(label);
    }

    btnGroupDiv.appendChild(btnGroup);
    container.appendChild(label);
    container.appendChild(btnGroupDiv);

    return container;
  }

function addRatingsToContainer(containerId) {
    const container = document.getElementById(containerId);
    const criteriaList = ['Correctness', 'Credibility'];

    criteriaList.forEach(criteria => {
        container.appendChild(createRatingSection(criteria, containerId));
    });
}

addRatingsToContainer('ratingContainerOmniQuery');
addRatingsToContainer('ratingContainerRAG');
