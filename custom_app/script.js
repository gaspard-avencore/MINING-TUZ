document.addEventListener('DOMContentLoaded', () => {
    const grid = document.querySelector('.grid');
    const generateGridButton = document.getElementById('generate-grid');
    const downloadCellsButton = document.getElementById('download-cells');
    const rowsInput = document.getElementById('rows');
    const columnsInput = document.getElementById('columns');

    let isMouseDown = false;
    let startHex = null;

    const setHexagonColor = (hex) => {
        hex.style.backgroundColor = 'red';
    };

    const clearHexagonColor = (hex) => {
        hex.style.backgroundColor = '#ccc';
    };

    const getHexPosition = (hex) => {
        return { row: parseInt(hex.dataset.row), col: parseInt(hex.dataset.col) };
    };

    const colorHexagonsInRectangle = (start, end) => {
        const [startRow, endRow] = [start.row, end.row].sort((a, b) => a - b);
        const [startCol, endCol] = [start.col, end.col].sort((a, b) => a - b);

        for (let row = startRow; row <= endRow; row++) {
            for (let col = startCol; col <= endCol; col++) {
                const hex = document.querySelector(`.hex[data-row='${row}'][data-col='${col}']`);
                setHexagonColor(hex);
            }
        }
    };

    generateGridButton.addEventListener('click', () => {
        const rows = parseInt(rowsInput.value);
        const columns = parseInt(columnsInput.value);

        // Calculate hexagon width and height based on the number of rows and columns
        const hexWidth = Math.min(window.innerWidth / (columns + 1), window.innerHeight / (rows * Math.sqrt(3) / 2 + 1));
        const hexHeight = hexWidth * Math.sqrt(3) / 2;
        const hexTransformY = hexHeight * 0.65; // 0.65 times the height of the hexagon
        const rowGap = hexHeight * 0.3; // 30% of the height for gap

        // Clear existing grid
        grid.innerHTML = '';

        // Update grid template columns and rows
        grid.style.gridTemplateColumns = `repeat(${columns}, ${hexWidth}px)`;
        grid.style.gridTemplateRows = `repeat(${rows}, ${hexHeight + rowGap}px)`; 

        for (let row = 0; row < rows; row++) {
            for (let col = 0; col < columns; col++) {
                const hex = document.createElement('div');
                hex.classList.add('hex');
                hex.style.width = `${hexWidth}px`;
                hex.style.height = `${hexHeight}px`;
                hex.style.gridColumn = col + 1;
                hex.style.gridRow = row + 1;
                hex.dataset.row = row;
                hex.dataset.col = col;
                if (col % 2 !== 0) {
                    hex.style.transform = `translateY(${hexTransformY}px)`; /* 0.65 times the height of the hexagon */
                }

                hex.addEventListener('mousedown', (e) => {
                    isMouseDown = true;
                    startHex = getHexPosition(hex);
                    if (e.ctrlKey || e.metaKey) {
                        // Ctrl or Command key pressed, clear the color
                        clearHexagonColor(hex);
                    } else {
                        setHexagonColor(hex);
                    }
                });

                hex.addEventListener('mousemove', (e) => {
                    if (isMouseDown && startHex) {
                        const currentHex = getHexPosition(hex);
                        colorHexagonsInRectangle(startHex, currentHex);
                    }
                });

                hex.addEventListener('click', (e) => {
                    if (!isMouseDown) {
                        if (e.ctrlKey || e.metaKey) {
                            // Ctrl or Command key pressed, clear the color
                            clearHexagonColor(hex);
                        } else {
                            setHexagonColor(hex);
                        }
                    }
                });

                grid.appendChild(hex);
            }
        }

        // Show the download cells button
        downloadCellsButton.style.display = 'block';
    });

    document.addEventListener('mouseup', () => {
        isMouseDown = false;
        startHex = null;
    });

    downloadCellsButton.addEventListener('click', () => {
        const rows = parseInt(rowsInput.value); // Get the number of rows
        const hexagons = document.querySelectorAll('.hex');
        const coloredHexagons = [];
        hexagons.forEach(hex => {
            if (hex.style.backgroundColor === 'red') {
                // Calculate the coordinates according to the new requirements
                const x = parseInt(hex.dataset.col);
                const y = rows - parseInt(hex.dataset.row) - 1; // Subtract 1 to make it 0-based
                coloredHexagons.push(`(${x}, ${y})`);
            }
        });
        const blob = new Blob([coloredHexagons.join(', ')], { type: 'text/plain' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'colored_hexagons.txt';
        a.click();
    });
});
