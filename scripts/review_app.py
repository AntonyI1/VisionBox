#!/usr/bin/env python3
"""
Review UI for detection captures.

Browse detections by class, approve or reject each.
Approved images move to the training dataset.

Run with: streamlit run scripts/review_app.py
"""

import streamlit as st
import json
import shutil
from pathlib import Path
from datetime import datetime
from PIL import Image

# Paths
CAPTURES_DIR = Path('captures/crops')
DATASET_DIR = Path('captures/dataset')
APPROVED_DIR = Path('datasets/training')
HISTORY_FILE = Path('datasets/review_history.json')

for d in [CAPTURES_DIR, APPROVED_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_history() -> list[dict]:
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_history(history: list[dict]):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def log_review(filename: str, class_name: str, action: str):
    history = load_history()
    history.append({
        'filename': filename,
        'class': class_name,
        'action': action,
        'reviewed_at': datetime.now().isoformat(),
    })
    save_history(history)


def get_classes() -> list[str]:
    """Get all class folders that have images."""
    if not CAPTURES_DIR.exists():
        return []
    classes = []
    for d in sorted(CAPTURES_DIR.iterdir()):
        if d.is_dir() and any(d.glob('*.jpg')):
            classes.append(d.name)
    return classes


def get_crops(class_name: str) -> list[Path]:
    """Get all crop images for a class."""
    class_dir = CAPTURES_DIR / class_name
    if not class_dir.exists():
        return []
    return sorted(class_dir.glob('*.jpg'))


def parse_crop_filename(path: Path) -> dict:
    """Extract track ID and confidence from filename.

    Format: track{id}_{timestamp}_{confidence}.jpg
    """
    stem = path.stem
    parts = stem.split('_')
    info = {'track': '?', 'confidence': '?', 'timestamp': ''}

    # Track ID is first part
    if parts and parts[0].startswith('track'):
        info['track'] = parts[0].replace('track', '#')

    # Confidence is last part (e.g., "0.85")
    try:
        info['confidence'] = f"{float(parts[-1]):.0%}"
    except (ValueError, IndexError):
        pass

    # Timestamp is everything in between
    if len(parts) >= 3:
        ts_parts = parts[1:-1]
        ts_str = '_'.join(ts_parts)
        try:
            dt = datetime.strptime(ts_str[:15], '%Y%m%d_%H%M%S')
            info['timestamp'] = dt.strftime('%b %d, %H:%M:%S')
        except (ValueError, IndexError):
            info['timestamp'] = ts_str

    return info


def approve_crop(path: Path, class_name: str):
    """Move crop to approved training folder."""
    dest_dir = APPROVED_DIR / class_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    shutil.move(str(path), str(dest))
    log_review(path.name, class_name, 'approved')


def reject_crop(path: Path, class_name: str):
    """Delete the crop."""
    path.unlink()
    log_review(path.name, class_name, 'rejected')


# --- Streamlit UI ---

st.set_page_config(page_title="VisionBox Review", layout="wide")

tab_review, tab_gallery, tab_history = st.tabs(["Review", "Gallery", "History"])

# ---- Review Tab (one-by-one) ----
with tab_review:
    st.header("Review by Class")

    classes = get_classes()

    if not classes:
        st.info(
            "No captures to review. Run detect_and_capture.py to collect detections."
        )
    else:
        # Class selector — persists across reruns so reject/approve
        # doesn't jump to a different class
        if 'selected_class' not in st.session_state:
            st.session_state.selected_class = classes[0]
        if st.session_state.selected_class not in classes:
            st.session_state.selected_class = classes[0]

        col_select, col_stats = st.columns([2, 3])
        with col_select:
            selected_class = st.selectbox(
                "Class",
                classes,
                index=classes.index(st.session_state.selected_class),
                format_func=lambda c: f"{c.replace('_', ' ')} ({len(get_crops(c))})",
                key="class_selector"
            )
            st.session_state.selected_class = selected_class
        with col_stats:
            st.write("")  # spacing
            for cls in classes:
                count = len(get_crops(cls))
                st.write(f"**{cls.replace('_', ' ')}**: {count} pending")

        crops = get_crops(selected_class)

        if not crops:
            st.success(f"All {selected_class} crops reviewed!")
        else:
            # Index management
            idx_key = f'idx_{selected_class}'
            if idx_key not in st.session_state:
                st.session_state[idx_key] = 0
            st.session_state[idx_key] = max(
                0, min(st.session_state[idx_key], len(crops) - 1)
            )

            current = crops[st.session_state[idx_key]]
            info = parse_crop_filename(current)

            # Progress
            st.progress((st.session_state[idx_key] + 1) / len(crops))
            st.write(
                f"**{st.session_state[idx_key] + 1} of {len(crops)}** | "
                f"Track {info['track']} | "
                f"Confidence: {info['confidence']} | "
                f"{info['timestamp']}"
            )

            # Fixed controls above image — buttons never move
            ctrl_left, ctrl_mid, ctrl_right = st.columns([1, 2, 1])
            with ctrl_left:
                if st.button(
                    "Reject", type="secondary", use_container_width=True,
                    key=f"reject_{selected_class}"
                ):
                    reject_crop(current, selected_class)
                    st.rerun()
            with ctrl_mid:
                nav1, nav2 = st.columns(2)
                with nav1:
                    if st.button(
                        "< Prev", use_container_width=True,
                        disabled=st.session_state[idx_key] == 0,
                        key=f"prev_{selected_class}"
                    ):
                        st.session_state[idx_key] -= 1
                        st.rerun()
                with nav2:
                    if st.button(
                        "Next >", use_container_width=True,
                        disabled=st.session_state[idx_key] >= len(crops) - 1,
                        key=f"next_{selected_class}"
                    ):
                        st.session_state[idx_key] += 1
                        st.rerun()
            with ctrl_right:
                if st.button(
                    "Approve", type="primary", use_container_width=True,
                    key=f"approve_{selected_class}"
                ):
                    approve_crop(current, selected_class)
                    st.rerun()

            # Show image at native resolution, capped at 600px wide
            img = Image.open(current)
            w, h = img.size
            display_w = min(w, 600)
            st.image(img, width=display_w,
                     caption=f"{w}x{h}px | {info['confidence']}")

    # Stats bar
    st.divider()
    s1, s2, s3 = st.columns(3)
    pending = sum(len(get_crops(c)) for c in classes) if classes else 0
    approved_count = sum(
        1 for _ in APPROVED_DIR.rglob('*.jpg')
    ) if APPROVED_DIR.exists() else 0
    history = load_history()
    s1.metric("Pending", pending)
    s2.metric("Approved", approved_count)
    s3.metric("Total Reviewed", len(history))


# ---- Gallery Tab (quick overview) ----
with tab_gallery:
    st.header("Capture Gallery")

    classes = get_classes()
    if not classes:
        st.info("No captures yet.")
    else:
        for cls in classes:
            crops = get_crops(cls)
            if not crops:
                continue

            display_name = cls.replace('_', ' ')
            st.subheader(f"{display_name} ({len(crops)})")

            # Show grid of thumbnails (max 20 per class)
            cols = st.columns(4)
            for i, crop_path in enumerate(crops[:20]):
                with cols[i % 4]:
                    info = parse_crop_filename(crop_path)
                    st.image(str(crop_path), caption=info['confidence'],
                             use_container_width=True)

            if len(crops) > 20:
                st.write(f"... and {len(crops) - 20} more")
            st.divider()


# ---- History Tab ----
with tab_history:
    st.header("Review History")

    history = load_history()

    if not history:
        st.info("No review history yet.")
    else:
        approved = [h for h in history if h['action'] == 'approved']
        rejected = [h for h in history if h['action'] == 'rejected']
        total = len(history)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Reviewed", total)
        s2.metric("Approved", len(approved))
        s3.metric("Rejected", len(rejected))
        s4.metric(
            "Approval Rate",
            f"{len(approved)/total:.0%}" if total > 0 else "N/A"
        )

        st.divider()

        # Filter
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            filter_action = st.selectbox(
                "Action", ["All", "Approved", "Rejected"]
            )
        with filter_col2:
            history_classes = sorted(set(h.get('class', '?') for h in history))
            filter_class = st.selectbox("Class", ["All"] + history_classes)

        filtered = history
        if filter_action == "Approved":
            filtered = [h for h in filtered if h['action'] == 'approved']
        elif filter_action == "Rejected":
            filtered = [h for h in filtered if h['action'] == 'rejected']

        if filter_class != "All":
            filtered = [h for h in filtered if h.get('class') == filter_class]

        for entry in reversed(filtered[-100:]):
            action_color = "green" if entry['action'] == 'approved' else "red"
            try:
                dt = datetime.fromisoformat(entry['reviewed_at'])
                time_str = dt.strftime("%b %d, %H:%M:%S")
            except (ValueError, TypeError, KeyError):
                time_str = "?"

            cls_display = entry.get('class', '?').replace('_', ' ')

            with st.container(border=True):
                c1, c2, c3 = st.columns([1, 3, 2])
                c1.write(f":{action_color}[**{entry['action'].upper()}**]")
                c2.write(f"**{cls_display}** | `{entry.get('filename', '?')}`")
                c3.write(time_str)
