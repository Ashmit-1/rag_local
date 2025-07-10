class RAGChatInterface {
    constructor() {
        this.selectedFiles = [];
        this.initializeElements();
        this.bindEvents();
    }

    initializeElements() {
        this.textInput = document.getElementById('textInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.addFileBtn = document.getElementById('addFileBtn');
        this.fileInput = document.getElementById('fileInput');
        this.fileList = document.getElementById('fileList');
        this.messagesArea = document.getElementById('messagesArea');
        this.uploadProgress = document.getElementById('uploadProgress');
    }

    bindEvents() {
        // Auto-resize textarea
        this.textInput.addEventListener('input', () => {
            this.textInput.style.height = 'auto';
            this.textInput.style.height = Math.min(this.textInput.scrollHeight, 120) + 'px';
        });

        // Send message on Enter (without Shift)
        this.textInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Send button click
        this.sendBtn.addEventListener('click', () => {
            this.sendMessage();
        });

        // Add file button click
        this.addFileBtn.addEventListener('click', () => {
            this.fileInput.click();
        });

        // File input change
        this.fileInput.addEventListener('change', (e) => {
            this.handleFileSelection(e.target.files);
        });
    }

    sendMessage() {
        const message = this.textInput.value.trim();
        if (!message) return;

        // Add user message
        this.addMessage(message, 'user');

        // Clear input
        this.textInput.value = '';
        this.textInput.style.height = 'auto';

        // Simulate assistant response
        setTimeout(() => {
            this.addMessage('I received your message. This is where the RAG system would process your query and provide a response based on the uploaded documents.', 'assistant');
        }, 1000);
    }

    addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        messageDiv.innerHTML = `<p>${text}</p>`;

        this.messagesArea.appendChild(messageDiv);
        this.messagesArea.scrollTop = this.messagesArea.scrollHeight;
    }

    handleFileSelection(files) {
        Array.from(files).forEach(file => {
            if (!this.selectedFiles.find(f => f.name === file.name)) {
                this.selectedFiles.push(file);
            }
        });

        this.updateFileList();
        this.uploadFilesToDatabase();
    }

    updateFileList() {
        this.fileList.innerHTML = '';

        this.selectedFiles.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';

            fileItem.innerHTML = `
                <span class="file-name">${file.name}</span>
                <span class="file-size">${this.formatFileSize(file.size)}</span>
                <button class="remove-file" data-index="${index}">×</button>
            `;

            const removeBtn = fileItem.querySelector('.remove-file');
            removeBtn.addEventListener('click', () => {
                this.removeFile(index);
            });

            this.fileList.appendChild(fileItem);
        });
    }

    removeFile(index) {
        this.selectedFiles.splice(index, 1);
        this.updateFileList();
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    uploadFilesToDatabase() {
        if (this.selectedFiles.length === 0) return;

        this.uploadProgress.textContent = 'Processing files for database...';

        // Simulate upload process
        setTimeout(() => {
            this.uploadProgress.textContent = `Successfully added ${this.selectedFiles.length} file(s) to the knowledge base.`;

            // Clear after a few seconds
            setTimeout(() => {
                this.uploadProgress.textContent = '';
                this.selectedFiles = [];
                this.updateFileList();
            }, 3000);
        }, 2000);
    }
}

// Initialize the chat interface
document.addEventListener('DOMContentLoaded', () => {
    new RAGChatInterface();
});