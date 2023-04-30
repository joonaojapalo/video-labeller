from pathlib import Path
import csv
from collections import defaultdict

import cv2


class Frame:
    def __init__(self, rel_frame, subject_id, trial_id, paths_by_cam) -> None:
        self.rel_frame = rel_frame
        self.subject_id = subject_id
        self.trial_id = trial_id
        self.paths_by_cam = paths_by_cam

    def get_image(self, cam_id):
        img_path = str(self.paths_by_cam[cam_id])
        return cv2.imread(img_path)


class ImageRepo:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.by_subject = self._read_subject_lookup(self.csv_path)

    def _row_dict(self, head, row):
        d = dict(zip(head, row))
        d["rel_frame"] = int(d["rel_frame"])
        return d

    def _read_subject_lookup(self, csv_path):
        reader = csv.reader(csv_path.open(), delimiter=";")
        head = next(reader)

        # build lookup by {subject_id: {trial_id: [...]}}
        subject_idx = head.index("subject_id")
        trial_idx = head.index("throw_id")
        event_idx = head.index("event_name")
        lookup = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    list
                )
            )
        )

        for row in reader:
            sid = row[subject_idx]
            tid = int(row[trial_idx])
            event_name = row[event_idx]
            lookup[sid][tid][event_name].append(self._row_dict(head, row))

        return lookup

    def get_cams(self) -> tuple:
        return ("oe", "ot")

    def get_subjects(self) -> list:
        # input_file;frame_file;subject_id;throw_id;cam_id;event_name;rel_frame;frame
        return sorted(self.by_subject.keys())

    def get_trials(self, subject_id) -> list:
        return sorted(self.by_subject[subject_id].keys())

    def _event_key(self, event):
        """Custom comparator function.
        """
        order = [
            "rltd",
            "bltd_first",
            "bltd",
            "release",
        ]
        return order.index(event) if event in order else len(order)

    def get_events(self, subject_id, trial_id) -> list:
        return sorted(self.by_subject[subject_id][trial_id].keys(),
                      key=self._event_key)

    def get_rel_frames(self, subject_id, trial_id, event_name, cam_id=None) -> list:
        frames = self.by_subject[subject_id][trial_id][event_name]

        # pick default cam_id
        if cam_id is None:
            cam_id = frames[0]["cam_id"]

        return sorted([f["rel_frame"] for f in frames if f["cam_id"] == cam_id])

    def get_frames(self, subject_id, trial_id) -> list:
        return self.by_subject[subject_id][trial_id]

    def _format_rel_frame(self, rel_frame):
        if rel_frame < 0:
            return str(rel_frame)
        elif rel_frame > 0:
            return f"+{rel_frame}"
        else:
            return "0"

    def get_frame(self, subject_id, trial_id, event_name, rel_frame) -> Frame:
        # find matching frame
        match = tuple(filter(
            lambda row: row["rel_frame"] == rel_frame,
            self.by_subject[subject_id][trial_id][event_name]
        ))

        if not match:
            return

        paths_by_cam = {}
        for frame in match:
            rel_frame_str = self._format_rel_frame(rel_frame)
            cam_id = frame["cam_id"]
            fname = f"{subject_id}_{trial_id}_{cam_id}_{event_name}_{rel_frame_str}.png"
            path = self.csv_path.parent.joinpath("frames").joinpath(fname)
            paths_by_cam[cam_id] = path

        # build path
        return Frame(int(frame["rel_frame"]),
                     frame["subject_id"],
                     frame["throw_id"],
                     paths_by_cam)
