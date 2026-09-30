import warnings

import numpy as np
import pandas as pd
import torch


def _is_blank(value):
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == '':
        return True
    return False


def _row_is_numeric(row_values, min_fraction=0.9):
    vals = [v for v in row_values if not _is_blank(v)]
    if len(vals) == 0:
        return False
    numeric = 0
    for v in vals:
        try:
            float(v)
            numeric += 1
        except (TypeError, ValueError):
            pass
    return numeric / len(vals) >= min_fraction


def detect_wavenumber_row(df):
    first_row = df.iloc[0].values
    if _is_blank(first_row[0]) and _row_is_numeric(first_row[1:]):
        return True, first_row
    return False, None


def _column_has_non_numeric(col_values):
    as_numeric = pd.to_numeric(pd.Series(col_values), errors='coerce')
    return bool(as_numeric.isna().any())


def read_spectral_excel(file_path, has_patient_id=None):
    df = pd.read_excel(file_path, header=None)

    has_wn, wavenumbers = detect_wavenumber_row(df)
    if has_wn:
        df = df.iloc[1:].reset_index(drop=True)

    n_rows = len(df)
    if n_rows == 0:
        raise ValueError(f"No data rows found in {file_path}")
    if df.shape[1] < 2:
        raise ValueError(
            f"{file_path}: expected at least 2 columns "
            f"(label + data), found {df.shape[1]}")

    labels = pd.to_numeric(df.iloc[:, 0], errors='raise').values.astype(np.int64)

    if has_patient_id is None:
       
        if _column_has_non_numeric(df.iloc[:, 1].values):
            has_patient_id = True
        else:
            raise ValueError(
                f"{file_path}: column 1 is purely numeric, so the file "
                "format is ambiguous (canonical with numeric patient ids "
                "vs. legacy without patient ids). Auto-detection is "
                "disabled to prevent silent column misinterpretation. "
                "Re-run with has_patient_id=True (canonical: col0=label, "
                "col1=patient_id, col2+=spectra) or has_patient_id=False "
                "(legacy: col0=label, col1+=spectra). For data_split.py "
                "use --input-format canonical|legacy."
            )

    if has_patient_id:
        if df.shape[1] < 3:
            raise ValueError(
                f"{file_path}: canonical format (has_patient_id=True) "
                f"requires >= 3 columns (label, patient_id, >=1 spectral "
                f"feature), found {df.shape[1]}")
        patient_ids = (df.iloc[:, 1]
                       .astype(str).str.strip()
                       .str.replace(r'\.0$', '', regex=True)
                       .values)
        spectra = df.iloc[:, 2:].apply(pd.to_numeric, errors='raise').values
        canonical = True
        if len(np.unique(patient_ids)) == n_rows:
            warnings.warn(
                f"{file_path}: every row has a distinct patient_id "
                f"({n_rows} unique ids / {n_rows} rows). This is valid for "
                "one-spectrum-per-patient data, but if these files were "
                "expected to contain technical replicates, verify that "
                "column 1 is really the patient id and not a spectral "
                "feature (wrong has_patient_id value?).",
                UserWarning,
            )
    else:
        patient_ids = None
        spectra = df.iloc[:, 1:].apply(pd.to_numeric, errors='raise').values
        canonical = False

    return {
        'labels': labels,
        'patient_ids': patient_ids,
        'spectra': spectra.astype(np.float32),
        'wavenumbers': wavenumbers,
        'canonical': canonical,
    }


def load_data(file_path, has_patient_id=None):

    parsed = read_spectral_excel(file_path, has_patient_id=has_patient_id)

    y_np = parsed['labels']
    X_np = parsed['spectra']

    if parsed['canonical']:
        groups = parsed['patient_ids']
    else:
        warnings.warn(
            f"{file_path}: no patient_id column detected (legacy format). "
            "Using row indices as groups; patient-level independence cannot "
            "be enforced. Re-run data_split.py to produce canonical files.",
            UserWarning,
        )
        groups = np.arange(len(y_np)).astype(str)

    X_tensor = torch.tensor(X_np, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y_np, dtype=torch.long)


    return X_tensor, y_tensor, groups


def write_spectral_excel(file_path, labels, patient_ids, spectra, wavenumbers=None):

    df = pd.DataFrame({'label': labels})
    if patient_ids is not None:
        df['patient_id'] = patient_ids
    spectra_df = pd.DataFrame(spectra)
    df = pd.concat([df, spectra_df], axis=1)
    df.columns = range(df.shape[1])

    if wavenumbers is not None:
        header_df = pd.DataFrame([list(wavenumbers)])
        header_df.columns = range(header_df.shape[1])
        df = pd.concat([header_df, df], ignore_index=True)

    df.to_excel(file_path, index=False, header=False)
