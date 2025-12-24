"""
Add MQTT log display to dashboard
"""

# Read the current dashboard.html
with open('dashboard.html', 'r') as f:
    content = f.read()

# Find the location to add the MQTT log (after stats row)
insert_point = content.find('<div class="row">\n        <!-- Main Dashboard -->')

# HTML for the new MQTT log section
mqtt_log_section = '''
        <!-- MQTT Log Row -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-dark text-white d-flex justify-content-between align-items-center">
                        <h5 class="mb-0"><i class="bi bi-activity me-2"></i>MQTT Log (Weight Events)</h5>
                        <button class="btn btn-sm btn-outline-light" id="clear-log-btn">Clear</button>
                    </div>
                    <div class="card-body p-2" id="mqtt-log" style="height: 200px; overflow-y: auto; background-color: #f8f9fa;">
                        <div class="text-center text-muted py-5">
                            <i class="bi bi-activity fs-1"></i>
                            <p>Waiting for weight events...</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
'''

# Insert the MQTT log section
if insert_point != -1:
    new_content = content[:insert_point] + mqtt_log_section + content[insert_point:]
    
    # Write the updated content back to the file
    with open('dashboard.html', 'w') as f:
        f.write(new_content)
    
    print("MQTT log section added to dashboard.html successfully!")
else:
    print("Could not find insertion point in dashboard.html")

print("\nNext step: Add MQTT log functionality to dashboard.js")