// Jupyter SQL Extension - Frontend JavaScript
// This file provides frontend integration for the SQL extension

define([
    'jquery',
    'base/js/namespace',
    'base/js/events'
], function($, Jupyter, events) {
    'use strict';

    // Extension initialization
    function load_ipython_extension() {
        console.log('Jupyter SQL Extension loaded');
        
        // Add CSS for better styling
        $('<style>')
            .prop('type', 'text/css')
            .html(`
                .sql-result {
                    margin: 10px 0;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
                .sql-result table {
                    width: 100%;
                    border-collapse: collapse;
                }
                .sql-result th, .sql-result td {
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                .sql-result th {
                    background-color: #f5f5f5;
                    font-weight: bold;
                }
                .sql-metadata {
                    padding: 8px;
                    background-color: #f9f9f9;
                    border-top: 1px solid #ddd;
                    font-size: 0.9em;
                    color: #666;
                }
                .sql-error {
                    border-left: 4px solid #d32f2f;
                    padding: 10px;
                    margin: 10px 0;
                    background-color: #f5f5f5;
                }
                .sql-warning {
                    border-left: 4px solid #f57c00;
                    padding: 10px;
                    margin: 10px 0;
                    background-color: #f5f5f5;
                }
            `)
            .appendTo('head');

        // Register the magic command
        if (Jupyter.notebook && Jupyter.notebook.kernel) {
            Jupyter.notebook.kernel.execute(`
                try:
                    %load_ext jupyter_sql_extension
                    print("✅ Jupyter SQL Extension loaded successfully!")
                    print("   Use %%sql to execute SQL queries.")
                except Exception as e:
                    print(f"⚠️  Failed to load SQL extension: {e}")
            `);
        }

        // Add menu item for SQL extension
        var menu = $('<li>')
            .append($('<a>')
                .attr('href', '#')
                .text('SQL Extension')
                .click(function(e) {
                    e.preventDefault();
                    show_sql_help();
                })
            );
        
        // Insert into help menu
        $('.dropdown-menu:contains("Help")').append(menu);
    }

    // Show SQL extension help
    function show_sql_help() {
        var help_content = `
            <div class="modal fade" id="sql-help-modal" tabindex="-1" role="dialog">
                <div class="modal-dialog modal-lg" role="document">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Jupyter SQL Extension Help</h5>
                            <button type="button" class="close" data-dismiss="modal">
                                <span>&times;</span>
                            </button>
                        </div>
                        <div class="modal-body">
                            <h6>Basic Usage:</h6>
                            <pre><code>%%sql your_connection_id
SELECT * FROM your_table LIMIT 10</code></pre>
                            
                            <h6>With Options:</h6>
                            <pre><code>%%sql your_db --format json --limit 5 --verbose
SELECT name, email FROM users WHERE active = true</code></pre>
                            
                            <h6>Variable Substitution:</h6>
                            <pre><code>user_id = 123
%%sql test_db
SELECT * FROM users WHERE id = {user_id}</code></pre>
                            
                            <h6>Available Options:</h6>
                            <ul>
                                <li><code>--format</code>: Output format (table, json, dataframe, html)</li>
                                <li><code>--limit</code>: Maximum number of rows</li>
                                <li><code>--timeout</code>: Query timeout in seconds</li>
                                <li><code>--no-cache</code>: Disable connection caching</li>
                                <li><code>--explain</code>: Show query execution plan</li>
                                <li><code>--dry-run</code>: Validate query without execution</li>
                                <li><code>--verbose</code>: Enable verbose output</li>
                            </ul>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        $('body').append(help_content);
        $('#sql-help-modal').modal('show');
        
        // Clean up modal after it's hidden
        $('#sql-help-modal').on('hidden.bs.modal', function() {
            $(this).remove();
        });
    }

    // Return the extension object
    return {
        load_ipython_extension: load_ipython_extension
    };
}); 