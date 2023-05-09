from pathlib import Path
import sqlite3
from typing import TypedDict, List

schema = """
CREATE TABLE landmarks (
    name TEXT PRIMARY KEY
);

CREATE TABLE events (
    name TEXT PRIMARY KEY
);

CREATE TABLE cameras (
    id TEXT PRIMARY KEY
);

INSERT INTO events VALUES
    ('rltd'),
    ('bltd0'),
    ('bltd'),
    ('release');

INSERT INTO landmarks VALUES
    ('RElbow'),
    ('LElbow'),
    ('RMidFinger'),
    ('LMidFinger'),
    ('RShoulder'),
    ('LShoulder'),
    ('RWrist'),
    ('LWrist');

INSERT INTO cameras VALUES
    ('oe'),
    ('ot');

CREATE TABLE markers (
    id INTEGER PRIMARY KEY,
    subject_id TEXT NOT NULL,
    trial_id INTEGER NOT NULL,
    event TEXT NOT NULL, 
    relative_frame INTEGER NOT NULL,
    cam_id TEXT NOT NULL,
    landmark TEXT NOT NULL,
    x REAL,
    y REAL,
    FOREIGN KEY(landmark) REFERENCES landmarks(name),
    FOREIGN KEY(event) REFERENCES events(name),
    FOREIGN KEY(cam_id) REFERENCES cameras(id),
    UNIQUE (subject_id, trial_id, event, relative_frame, cam_id, landmark)
);
"""

query_select_frame = """
SELECT
    *
FROM
    markers
WHERE
    subject_id = ? AND
    trial_id = ? AND
    relative_frame = ?;
"""

query_insert_landmark = """
INSERT INTO markers (
    subject_id,
    trial_id,
    event,
    relative_frame,
    cam_id,
    landmark,
    x,
    y
)
VALUES (
    ?, ?, ?, ?, ?, ?, ?, ?
);
"""

query_update_landmark = """
UPDATE markers
SET
    x = ?,
    y = ?
WHERE
    id = ?
"""

query_select_landmarks = "select name from landmarks;"

class Marker (TypedDict):
    id: int
    subject_id: str
    trial_id: str
    event: str
    relative_frame: int
    cam_id: str
    landmark: str
    x: float
    y: float


def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


class SQLiteLabelRepo:
    def __init__(self, dbfile):
        dbpath = Path(dbfile)

        if not dbpath.is_file():
            self.conn = self.init_sqlite_db(dbpath)
        else:
            self.conn = self._connect(dbpath)

    def __del__(self):
        self.conn.close()

    def _connect(self, dbpath):
        conn = sqlite3.connect(
            str(dbpath), detect_types=sqlite3.PARSE_COLNAMES)
        conn.row_factory = dict_factory
        return conn

    def init_sqlite_db(self, dbpath):
        conn = self._connect(dbpath)
        with conn:
            for query in schema.split(";"):
                conn.execute(f"{query};")

        return conn

    def get_available_landmarks(self) -> List[str]:
        cur = self.conn.cursor()
        res = cur.execute(query_select_landmarks)
        return [row["name"] for row in res.fetchall()]
        
    def get_frame(self, subject_id: str, trial_id: int, relative_frame: int) -> List[Marker]:
        cur = self.conn.cursor()
        params = (
            subject_id,
            trial_id,
            relative_frame,
        )
        res = cur.execute(query_select_frame, params)
        return res.fetchall()

    def update_point(self, point_id, x, y):
        with self.conn:
            params = (
                x,
                y,
                point_id,
            )
            self.conn.execute(query_update_landmark, params)

    def create_point(self, subject_id, trial_id, event_name, relative_frame, cam_id, landmark, x, y):
        with self.conn:
            params = (
                subject_id,
                trial_id,
                event_name,
                relative_frame,
                cam_id,
                landmark,
                x,
                y,
            )
            self.conn.execute(query_insert_landmark, params)


if __name__ == "__main__":
    r = SQLiteLabelRepo("test.db3")
    points = r.get_frame("S101", 1, -8, "oe")

    if not points:
        r.create_point("S101", 1, "bltd0", -8, "oe", "RElbow", 101.56, 90.2)
    else:
        point = points[0]
        r.update_point(point["id"], 823.4, 232)

    print(r.get_frame("S101", 1, -8))
