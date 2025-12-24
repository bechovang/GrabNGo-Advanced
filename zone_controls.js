// Zone Controls JavaScript for Dashboard

// Load zone settings from server and update UI
function loadZoneSettings() {
    fetch('/dashboard/zones')
        .then(response => response.json())
        .then(data => {
            if (data.qr_zone) {
                document.getElementById('qr-zone-x').value = data.qr_zone.x1_percent * 100;
                document.getElementById('qr-zone-y').value = data.qr_zone.y1_percent * 100;
                document.getElementById('qr-zone-width').value = (data.qr_zone.x2_percent - data.qr_zone.x1_percent) * 100;
                document.getElementById('qr-zone-height').value = (data.qr_zone.y2_percent - data.qr_zone.y1_percent) * 100;
                
                updateRangeValue('qr-zone-x');
                updateRangeValue('qr-zone-y');
                updateRangeValue('qr-zone-width');
                updateRangeValue('qr-zone-height');
            }
            
            if (data.shelf_zone) {
                document.getElementById('shelf-zone-x').value = data.shelf_zone.x1_percent * 100;
                document.getElementById('shelf-zone-y').value = data.shelf_zone.y1_percent * 100;
                document.getElementById('shelf-zone-width').value = (data.shelf_zone.x2_percent - data.shelf_zone.x1_percent) * 100;
                document.getElementById('shelf-zone-height').value = (data.shelf_zone.y2_percent - data.shelf_zone.y1_percent) * 100;
                
                updateRangeValue('shelf-zone-x');
                updateRangeValue('shelf-zone-y');
                updateRangeValue('shelf-zone-width');
                updateRangeValue('shelf-zone-height');
            }
        })
        .catch(error => {
            console.error('Error loading zone settings:', error);
        });
}

// Open zone settings modal
function openZoneSettings() {
    const modal = new bootstrap.Modal(document.getElementById('zoneSettingsModal'));
    modal.show();
}

// Save zone settings to server
function saveZoneSettings() {
    const qrZone = {
        x1_percent: document.getElementById('qr-zone-x').value / 100,
        y1_percent: document.getElementById('qr-zone-y').value / 100,
        x2_percent: (document.getElementById('qr-zone-x').value / 100) + (document.getElementById('qr-zone-width').value / 100),
        y2_percent: (document.getElementById('qr-zone-y').value / 100) + (document.getElementById('qr-zone-height').value / 100)
    };
    
    const shelfZone = {
        x1_percent: document.getElementById('shelf-zone-x').value / 100,
        y1_percent: document.getElementById('shelf-zone-y').value / 100,
        x2_percent: (document.getElementById('shelf-zone-x').value / 100) + (document.getElementById('shelf-zone-width').value / 100),
        y2_percent: (document.getElementById('shelf-zone-y').value / 100) + (document.getElementById('shelf-zone-height').value / 100)
    };
    
    const zoneData = {
        qr_zone: qrZone,
        shelf_zone: shelfZone
    };
    
    fetch('/dashboard/zones', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(zoneData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Show success message
            alert('Zone settings saved successfully!');
            
            // Refresh dashboard data to apply new zones
            pollData();
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('zoneSettingsModal'));
            modal.hide();
        } else {
            alert('Error saving zone settings: ' + (data.error || 'Unknown error'));
        }
    })
    .catch(error => {
        console.error('Error saving zone settings:', error);
        alert('Error saving zone settings. Please try again.');
    });
}

// Setup range input handlers
function setupRangeInputs() {
    // QR Zone inputs
    document.getElementById('qr-zone-x').addEventListener('input', function() {
        updateRangeValue('qr-zone-x');
    });
    
    document.getElementById('qr-zone-y').addEventListener('input', function() {
        updateRangeValue('qr-zone-y');
    });
    
    document.getElementById('qr-zone-width').addEventListener('input', function() {
        updateRangeValue('qr-zone-width');
    });
    
    document.getElementById('qr-zone-height').addEventListener('input', function() {
        updateRangeValue('qr-zone-height');
    });
    
    // Shelf Zone inputs
    document.getElementById('shelf-zone-x').addEventListener('input', function() {
        updateRangeValue('shelf-zone-x');
    });
    
    document.getElementById('shelf-zone-y').addEventListener('input', function() {
        updateRangeValue('shelf-zone-y');
    });
    
    document.getElementById('shelf-zone-width').addEventListener('input', function() {
        updateRangeValue('shelf-zone-width');
    });
    
    document.getElementById('shelf-zone-height').addEventListener('input', function() {
        updateRangeValue('shelf-zone-height');
    });
}

// Update range value display
function updateRangeValue(inputId) {
    const input = document.getElementById(inputId);
    const valueDisplay = document.getElementById(inputId + '-value');
    if (input && valueDisplay) {
        valueDisplay.textContent = input.value + '%';
    }
}

// Update zone visualization on map
function updateZoneVisualization(zones) {
    if (!zones) return;
    
    // Update QR zone
    if (zones.qr_zone) {
        const qrZone = document.querySelector('.qr-zone');
        if (qrZone) {
            const x = zones.qr_zone.x1_percent * 100;
            const y = zones.qr_zone.y1_percent * 100;
            const width = (zones.qr_zone.x2_percent - zones.qr_zone.x1_percent) * 100;
            const height = (zones.qr_zone.y2_percent - zones.qr_zone.y1_percent) * 100;
            
            qrZone.style.left = x + '%';
            qrZone.style.top = y + '%';
            qrZone.style.width = width + '%';
            qrZone.style.height = height + '%';
        }
    }
    
    // Update Shelf zone
    if (zones.shelf_zone) {
        const shelfZone = document.querySelector('.shelf-zone');
        if (shelfZone) {
            const x = zones.shelf_zone.x1_percent * 100;
            const y = zones.shelf_zone.y1_percent * 100;
            const width = (zones.shelf_zone.x2_percent - zones.shelf_zone.x1_percent) * 100;
            const height = (zones.shelf_zone.y2_percent - zones.shelf_zone.y1_percent) * 100;
            
            shelfZone.style.left = x + '%';
            shelfZone.style.top = y + '%';
            shelfZone.style.width = width + '%';
            shelfZone.style.height = height + '%';
        }
    }
}

// Add these functions to the existing dashboard.js
// They will be called when the dashboard loads