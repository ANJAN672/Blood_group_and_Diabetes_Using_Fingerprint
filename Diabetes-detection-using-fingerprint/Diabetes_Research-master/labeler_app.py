import os
from pathlib import Path
import streamlit as st
from PIL import Image
import pandas as pd


def get_image_list(data_dir):
    p = Path(data_dir)
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    files = [str(x) for x in p.rglob('*') if x.suffix.lower() in exts and x.is_file()]
    files.sort()
    return files


def save_label(csv_path, img_path, label):
    df = pd.DataFrame([[img_path, label]], columns=['path', 'label'])
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        # avoid duplicate entries
        if img_path in df_existing['path'].values:
            df_existing.loc[df_existing['path'] == img_path, 'label'] = label
            df_existing.to_csv(csv_path, index=False)
            return
        df_existing = pd.concat([df_existing, df], ignore_index=True)
        df_existing.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, index=False)


def main():
    st.title('Fingerprint — Diabetes Labeler')
    data_path = st.text_input('Dataset path', value=os.environ.get('DIABETES_DATA_PATH', 'dataset'))
    csv_path = st.text_input('Labels CSV path', value='labels/diabetes_labels.csv')

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if not os.path.exists(data_path):
        st.warning(f'Dataset path not found: {data_path}\nPlease copy your dataset into this workspace under the folder and retry.')
        return

    files = get_image_list(data_path)
    if not files:
        st.warning('No image files found under dataset path.')
        return

    if 'index' not in st.session_state:
        st.session_state.index = 0

    idx = st.session_state.index
    st.sidebar.write(f'Image {idx+1} / {len(files)}')

    cols = st.columns([1, 3])
    with cols[0]:
        if st.button('Prev'):
            st.session_state.index = max(0, st.session_state.index - 1)
            st.experimental_rerun()
        if st.button('Next'):
            st.session_state.index = min(len(files) - 1, st.session_state.index + 1)
            st.experimental_rerun()

    img_path = files[idx]
    st.write(img_path)
    img = Image.open(img_path).convert('RGB')
    st.image(img, use_column_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button('Label: Non-diabetic'):
            save_label(csv_path, img_path, 'non_diabetic')
            st.session_state.index = min(len(files) - 1, st.session_state.index + 1)
            st.experimental_rerun()
    with col2:
        if st.button('Label: Diabetic'):
            save_label(csv_path, img_path, 'diabetic')
            st.session_state.index = min(len(files) - 1, st.session_state.index + 1)
            st.experimental_rerun()
    with col3:
        if st.button('Skip'):
            st.session_state.index = min(len(files) - 1, st.session_state.index + 1)
            st.experimental_rerun()

    # show a small preview of labels
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.write('Labeled count:', len(df))
        st.dataframe(df.tail(10))


if __name__ == '__main__':
    main()
