


import openpyxl
import io
import tempfile
import pandas as pd




def process_excel(uploaded_file):
        # Load the workbook
        wb = openpyxl.load_workbook(filename=uploaded_file)
        ws = wb.active

        # Iterate through the cells, skipping the first row and the first column
        for row in ws.iter_rows(min_row=2, min_col=2):
            for cell in row:
                # Set the cell value to the background color's hex code
                cell.value = cell.fill.start_color.index[2:] if cell.fill.start_color.index else None

        # Save the updated workbook to a temporary file and return it
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
            wb.save(tmp.name)
            tmp.seek(0)
            return pd.read_excel(io.BytesIO(tmp.read()))