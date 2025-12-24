"""
Add zone controls to dashboard
"""

# Read current dashboard.html
with open('dashboard.html', 'r') as f:
    content = f.read()

# Find location to add zone controls (after time display)
insert_point = content.find('<div class="col-md-4 text-end">')

# HTML for zone controls
zone_controls = '''
                    <!-- Zone Controls -->
                    <div class="system-status">
                        <div class="me-3">
                            <button class="btn btn-sm btn-outline-light" id="zone-settings-btn">
                                <i class="bi bi-gear"></i> Zones
                            </button>
                        </div>
                        <div>
                            <span class="status-indicator" id="connection-status"></span>
                            <span id="connection-text">Connected</span>
                        </div>
                    </div>

<!-- Zone Settings Modal -->
<div class="modal fade" id="zoneSettingsModal" tabindex="-1" aria-labelledby="zoneSettingsModalLabel" aria-hidden="true">
  <div class="modal-dialog">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title" id="zoneSettingsModalLabel">
          <i class="bi bi-bounding-box me-2"></i>Zone Settings
        </h5>
        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
      </div>
      <div class="modal-body">
        <ul class="nav nav-tabs" id="zoneTabs" role="tablist">
          <li class="nav-item" role="presentation">
            <button class="nav-link active" id="qr-tab" data-bs-toggle="tab" data-bs-target="#qr-zone-tab" type="button" role="tab" aria-controls="qr-zone-tab" aria-selected="true">QR Zone</button>
          </li>
          <li class="nav-item" role="presentation">
            <button class="nav-link" id="shelf-tab" data-bs-toggle="tab" data-bs-target="#shelf-zone-tab" type="button" role="tab" aria-controls="shelf-zone-tab" aria-selected="false">Shelf Zone</button>
          </li>
        </ul>
        <div class="tab-content" id="zoneTabContent">
          <!-- QR Zone Settings -->
          <div class="tab-pane fade show active" id="qr-zone-tab" role="tabpanel" aria-labelledby="qr-tab">
            <div class="mb-3">
              <h6>QR Zone Position & Size</h6>
              <div class="row">
                <div class="col-md-6">
                  <div class="mb-3">
                    <label for="qr-zone-x" class="form-label">X Position (%)</label>
                    <input type="range" class="form-range" min="0" max="100" id="qr-zone-x" value="70">
                    <div class="d-flex justify-content-between">
                      <small>0%</small>
                      <span id="qr-zone-x-value">70%</span>
                      <small>100%</small>
                    </div>
                  </div>
                </div>
                <div class="col-md-6">
                  <div class="mb-3">
                    <label for="qr-zone-y" class="form-label">Y Position (%)</label>
                    <input type="range" class="form-range" min="0" max="100" id="qr-zone-y" value="0">
                    <div class="d-flex justify-content-between">
                      <small>0%</small>
                      <span id="qr-zone-y-value">0%</span>
                      <small>100%</small>
                    </div>
                  </div>
                </div>
                <div class="row">
                  <div class="col-md-6">
                    <div class="mb-3">
                      <label for="qr-zone-width" class="form-label">Width (%)</label>
                      <input type="range" class="form-range" min="10" max="100" id="qr-zone-width" value="30">
                      <div class="d-flex justify-content-between">
                        <small>10%</small>
                        <span id="qr-zone-width-value">30%</span>
                        <small>100%</small>
                      </div>
                    </div>
                  </div>
                  <div class="col-md-6">
                    <div class="mb-3">
                      <label for="qr-zone-height" class="form-label">Height (%)</label>
                      <input type="range" class="form-range" min="10" max="100" id="qr-zone-height" value="100">
                      <div class="d-flex justify-content-between">
                        <small>10%</small>
                        <span id="qr-zone-height-value">100%</span>
                        <small>100%</small>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <!-- Shelf Zone Settings -->
          <div class="tab-pane fade" id="shelf-zone-tab" role="tabpanel" aria-labelledby="shelf-tab">
            <div class="mb-3">
              <h6>Shelf Zone Position & Size</h6>
              <div class="row">
                <div class="col-md-6">
                  <div class="mb-3">
                    <label for="shelf-zone-x" class="form-label">X Position (%)</label>
                    <input type="range" class="form-range" min="0" max="100" id="shelf-zone-x" value="5">
                    <div class="d-flex justify-content-between">
                      <small>0%</small>
                      <span id="shelf-zone-x-value">5%</span>
                      <small>100%</small>
                    </div>
                  </div>
                </div>
                <div class="col-md-6">
                  <div class="mb-3">
                    <label for="shelf-zone-y" class="form-label">Y Position (%)</label>
                    <input type="range" class="form-range" min="0" max="100" id="shelf-zone-y" value="30">
                    <div class="d-flex justify-content-between">
                      <small>0%</small>
                      <span id="shelf-zone-y-value">30%</span>
                      <small>100%</small>
                    </div>
                  </div>
                </div>
                <div class="row">
                  <div class="col-md-6">
                    <div class="mb-3">
                      <label for="shelf-zone-width" class="form-label">Width (%)</label>
                      <input type="range" class="form-range" min="10" max="100" id="shelf-zone-width" value="45">
                      <div class="d-flex justify-content-between">
                        <small>10%</small>
                        <span id="shelf-zone-width-value">45%</span>
                        <small>100%</small>
                      </div>
                    </div>
                  </div>
                  <div class="col-md-6">
                    <div class="mb-3">
                      <label for="shelf-zone-height" class="form-label">Height (%)</label>
                      <input type="range" class="form-range" min="10" max="100" id="shelf-zone-height" value="60">
                      <div class="d-flex justify-content-between">
                        <small>10%</small>
                        <span id="shelf-zone-height-value">60%</span>
                        <small>100%</small>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div class="alert alert-info mt-3">
          <h6><i class="bi bi-info-circle me-2"></i>Zone Controls</h6>
          <p class="mb-1">• Adjust zone positions and sizes to match your camera view</p>
          <p class="mb-1">• Zones are defined as percentages of the camera view</p>
          <p class="mb-0">• Changes are applied immediately and saved for future sessions</p>
        </div>
      </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
        <button type="button" class="btn btn-primary" id="save-zone-settings">Save Settings</button>
      </div>
    </div>
  </div>
</div>
'''

# Insert zone controls and modal
if insert_point != -1:
    # Find the end of the div
    end_div = content.find('</div>', insert_point)
    if end_div != -1:
        new_content = content[:end_div+6] + zone_controls + content[end_div+6:]
        
        # Write updated content back to file
        with open('dashboard.html', 'w') as f:
            f.write(new_content)
        
        print("Zone controls added to dashboard.html successfully!")
    else:
        print("Could not find the end of the div in dashboard.html")
else:
    print("Could not find insertion point for zone controls in dashboard.html")

print("\nNext step: Add zone control functionality to dashboard.js")