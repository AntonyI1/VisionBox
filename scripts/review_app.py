#!/usr/bin/env python3
"""
Streamlit app for reviewing auto-captured detections.

This is the human-in-the-loop part of the active learning pipeline:
1. Model captures high-confidence detections
2. Human reviews them here (Correct/Incorrect)
3. Correct ones go to training data
4. Retrain model with expanded dataset
5. Repeat - this is the "data flywheel"

Run with: streamlit run scripts/review_app.py
"""

import streamlit as st
import json
import shutil
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw

# Paths
REVIEW_DIR = Path('datasets/review')
TRAIN_IMAGES = Path('datasets/bottles/images')
TRAIN_LABELS = Path('datasets/bottles/labels')
HISTORY_FILE = Path('datasets/review_history.json')

# Ensure directories exist
REVIEW_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_IMAGES.mkdir(parents=True, exist_ok=True)
TRAIN_LABELS.mkdir(parents=True, exist_ok=True)


def load_history() -> list[dict]:
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_history(history: list[dict]):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def log_review(name: str, action: str, detections: list[dict]):
    """Log a review decision to history."""
    history = load_history()
    history.append({
        'name': name,
        'action': action,
        'detections': detections,
        'reviewed_at': datetime.now().isoformat(),
    })
    save_history(history)


def get_review_items():
    """Get all items pending review (images with matching JSON)."""
    if not REVIEW_DIR.exists():
        return []

    items = []
    for json_path in sorted(REVIEW_DIR.glob('*.json')):
        img_path = json_path.with_suffix('.jpg')
        label_path = json_path.with_suffix('.txt')

        if img_path.exists():
            items.append({
                'name': json_path.stem,
                'image': img_path,
                'json': json_path,
                'label': label_path if label_path.exists() else None
            })
    return items


def draw_boxes(image: Image.Image, detections: list) -> Image.Image:
    """Draw bounding boxes on image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for det in detections:
        box = det['box']
        x1, y1, x2, y2 = box
        conf = det['confidence']
        cls = det['class']

        draw.rectangle([x1, y1, x2, y2], outline='lime', width=3)
        label = f"{cls} {conf:.0%}"
        draw.text((x1, y1 - 15), label, fill='lime')

    return img


def approve_item(item: dict, detections: list[dict]):
    """Move item to training dataset and log it."""
    dest_img = TRAIN_IMAGES / item['image'].name
    shutil.move(str(item['image']), str(dest_img))

    if item['label'] and item['label'].exists():
        dest_label = TRAIN_LABELS / item['label'].name
        shutil.move(str(item['label']), str(dest_label))

    item['json'].unlink()
    log_review(item['name'], 'accepted', detections)


def reject_item(item: dict, detections: list[dict]):
    """Delete item and log it."""
    item['image'].unlink()
    item['json'].unlink()
    if item['label'] and item['label'].exists():
        item['label'].unlink()

    log_review(item['name'], 'rejected', detections)


# --- Streamlit UI ---

st.set_page_config(page_title="VisionBox Review", layout="wide")

tab_review, tab_history = st.tabs(["Review", "History"])

# ---- Review Tab ----
with tab_review:
    st.header("Detection Review")

    items = get_review_items()

    if not items:
        st.info("No items to review. Run capture_for_review.py to collect samples.")
    else:
        if 'idx' not in st.session_state:
            st.session_state.idx = 0

        st.session_state.idx = max(0, min(st.session_state.idx, len(items) - 1))
        current = items[st.session_state.idx]

        st.progress(st.session_state.idx / len(items))
        st.write(f"**{st.session_state.idx + 1} of {len(items)}** | `{current['name']}`")

        col1, col2 = st.columns([3, 1])

        with col1:
            image = Image.open(current['image'])
            with open(current['json']) as f:
                meta = json.load(f)
            annotated = draw_boxes(image, meta['detections'])
            st.image(annotated, use_container_width=True)

        with col2:
            st.subheader("Detections")
            for det in meta['detections']:
                st.write(f"- **{det['class']}**: {det['confidence']:.1%}")

            st.divider()

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Correct", type="primary", use_container_width=True):
                    approve_item(current, meta['detections'])
                    st.rerun()
            with c2:
                if st.button("Incorrect", type="secondary", use_container_width=True):
                    reject_item(current, meta['detections'])
                    st.rerun()

            st.divider()

            nav1, nav2 = st.columns(2)
            with nav1:
                if st.button("< Prev", disabled=st.session_state.idx == 0):
                    st.session_state.idx -= 1
                    st.rerun()
            with nav2:
                if st.button("Next >", disabled=st.session_state.idx >= len(items) - 1):
                    st.session_state.idx += 1
                    st.rerun()

    # Quick stats at bottom
    st.divider()
    s1, s2, s3 = st.columns(3)
    train_count = len(list(TRAIN_IMAGES.glob('*')))
    review_count = len(items) if items else 0
    history = load_history()
    s1.metric("Training Images", train_count)
    s2.metric("Pending Review", review_count)
    s3.metric("Total Reviewed", len(history))

# ---- History Tab ----
with tab_history:
    st.header("Review History")

    history = load_history()

    if not history:
        st.info("No review history yet. Start reviewing to build history.")
    else:
        # Summary stats
        accepted = [h for h in history if h['action'] == 'accepted']
        rejected = [h for h in history if h['action'] == 'rejected']
        total = len(history)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Reviewed", total)
        s2.metric("Accepted", len(accepted))
        s3.metric("Rejected", len(rejected))
        s4.metric("Acceptance Rate", f"{len(accepted)/total:.0%}" if total > 0 else "N/A")

        st.divider()

        # Filter
        filter_action = st.selectbox("Filter", ["All", "Accepted", "Rejected"])
        if filter_action == "Accepted":
            filtered = accepted
        elif filter_action == "Rejected":
            filtered = rejected
        else:
            filtered = history

        # Show history (most recent first)
        for entry in reversed(filtered):
            action_icon = "+" if entry['action'] == 'accepted' else "x"
            action_color = "green" if entry['action'] == 'accepted' else "red"
            timestamp = entry.get('reviewed_at', 'unknown')

            # Parse timestamp for display
            try:
                dt = datetime.fromisoformat(timestamp)
                time_str = dt.strftime("%b %d, %H:%M:%S")
            except (ValueError, TypeError):
                time_str = timestamp

            classes = ", ".join(
                f"{d['class']} ({d['confidence']:.0%})"
                for d in entry.get('detections', [])
            )

            with st.container(border=True):
                c1, c2, c3 = st.columns([1, 3, 2])
                c1.write(f":{action_color}[**{entry['action'].upper()}**]")
                c2.write(f"`{entry['name']}` - {classes}")
                c3.write(f"{time_str}")
