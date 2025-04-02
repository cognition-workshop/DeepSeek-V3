let attentionData = [];
let currentLayer = 0;
let currentHead = 0;
let numLayers = 0;
let numHeads = 0;

$(document).ready(function() {
    $('#add-sample').click(function() {
        const sampleCount = $('#samples .sample-container').length + 1;
        const newSample = `
            <div class="sample-container">
                <div class="mb-3">
                    <label for="sample${sampleCount}" class="form-label">Sample ${sampleCount}</label>
                    <textarea id="sample${sampleCount}" class="form-control" rows="3"></textarea>
                </div>
                <button class="remove-sample btn btn-outline-danger btn-sm">Remove</button>
            </div>
        `;
        $('#samples').append(newSample);
    });
    
    $(document).on('click', '.remove-sample', function() {
        $(this).closest('.sample-container').remove();
        $('#samples .sample-container').each(function(idx) {
            $(this).find('.form-label').text(`Sample ${idx+1}`);
            $(this).find('textarea').attr('id', `sample${idx+1}`);
        });
    });
    
    $('#visualize').click(function() {
        $('#loading').show();
        $('#visualization').hide();
        
        const texts = [];
        $('#samples .sample-container textarea').each(function() {
            const text = $(this).val().trim();
            if (text) {
                texts.push(text);
            }
        });
        
        if (texts.length === 0) {
            alert('Please enter at least one text sample.');
            $('#loading').hide();
            return;
        }
        
        $.ajax({
            url: '/extract_attention',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ texts: texts }),
            success: function(data) {
                attentionData = data.results;
                if (attentionData.length > 0) {
                    numLayers = attentionData[0].attention_weights.length;
                    numHeads = attentionData[0].attention_weights[0].weights.shape[1];
                    
                    populateSelectors();
                    
                    visualizeAttention();
                    
                    $('#visualization').show();
                }
                $('#loading').hide();
            },
            error: function(error) {
                console.error('Error extracting attention:', error);
                alert('Error extracting attention patterns. Please try again.');
                $('#loading').hide();
            }
        });
    });
    
    $(document).on('change', '#layer-select', function() {
        currentLayer = parseInt($(this).val());
        visualizeAttention();
    });
    
    $(document).on('change', '#head-select', function() {
        currentHead = parseInt($(this).val());
        visualizeAttention();
    });
});

function populateSelectors() {
    $('#layer-select').empty();
    $('#head-select').empty();
    
    for (let i = 0; i < numLayers; i++) {
        $('#layer-select').append(`<option value="${i}">Layer ${i+1}</option>`);
    }
    
    for (let i = 0; i < numHeads; i++) {
        $('#head-select').append(`<option value="${i}">Head ${i+1}</option>`);
    }
    
    currentLayer = 0;
    currentHead = 0;
}

function visualizeAttention() {
    $('#attention-heatmaps').empty();
    
    attentionData.forEach((sample, idx) => {
        const tokens = sample.tokens;
        const weights = sample.attention_weights[currentLayer].weights;
        
        const container = $(`<div id="heatmap-${idx}" class="attention-heatmap"></div>`);
        $('#attention-heatmaps').append(container);
        
        const headWeights = weights.slice([null, currentHead, null]).squeeze();
        
        const heatmapData = [{
            z: headWeights,
            x: tokens,
            y: tokens,
            type: 'heatmap',
            colorscale: 'Viridis',
            hoverongaps: false,
            colorbar: {
                title: 'Attention Weight',
                thickness: 15,
            }
        }];
        
        const layout = {
            title: `Sample ${idx+1}`,
            xaxis: {
                title: 'Target Tokens',
                tickangle: -45
            },
            yaxis: {
                title: 'Source Tokens'
            },
            height: 500,
            margin: {
                l: 100,
                r: 50,
                b: 100,
                t: 50,
                pad: 4
            }
        };
        
        Plotly.newPlot(`heatmap-${idx}`, heatmapData, layout);
    });
}
