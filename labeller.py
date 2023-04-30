import sys
import argparse
import matplotlib.pyplot as plt
import json
from BlittedCursor import BlittedCursor

from ImageRepo import ImageRepo
from SQLiteLabelRepo import SQLiteLabelRepo


def scale(ax, scale, center_x, center_y, ref_height=1080):
    dx = (ref_height / 2) / scale
    ax.set_ylim(center_y + dx, center_y - dx)
    ax.set_xlim(center_x - dx, center_x + dx)


class LabellerApp:
    is_drawing = False

    def __init__(self,
                 image_repo: ImageRepo,
                 repo: SQLiteLabelRepo,
                 subject_id: str,
                 trial_id: int,
                 cam_ids=["oe", "ot"]) -> None:
        self.image_repo = image_repo
        self.repo = repo
        self.subject_id = subject_id
        self.trial_id = trial_id
        self.avail_landmarks = repo.get_available_landmarks()

        # event names
        self.event_names = image_repo.get_events(subject_id, trial_id)

        # setup subpolots
        self.fig, ax = plt.subplots(1, 2)
        self.ax1 = plt.subplot(1, 2, 1)
        self.ax2 = plt.subplot(1, 2, 2)
        self.axes = [self.ax1, self.ax2]
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1)

        # crosshair
        self.cursor1 = BlittedCursor(self.ax1)

        self.fig.canvas.mpl_connect('key_press_event', self.onkeypress)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('motion_notify_event',
                                    self.cursor1.on_mouse_move
                                    )

        self.im1 = None
        self.im2 = None

        # init state
        self.i_kp = 0
        self.i_event = 0
        self.i_frame = 0
        self.i_cam = 0
        self.cam_ids = cam_ids
        self.markers = None

        # bootstrap render
        self.load_markers()
        self.title = self.fig.suptitle(self._get_title())
        self.marker_line = self._draw_initial_landmark_object()
        _cam_ids = self._current_cam_ids()
        self.ax1.set_xlabel(f"cam: {_cam_ids[0]}")
        self.ax2.set_xlabel(f"cam: {_cam_ids[1]}")
        self.load_image(*self._get_image_paths())
        print(" * Ready!")

    def load_markers(self):
        print("Current relative frame:", self._current_rel_frame())
        self.markers = self.repo.get_frame(self.subject_id,
                                           self.trial_id,
                                           self._current_rel_frame()
                                           )

    def _get_title(self):
        return f"{self.avail_landmarks[self.i_kp]} ({self.subject_id}/{self.trial_id}, {self._current_event_name()}, frame={self._current_rel_frame()})"

    def _current_event_name(self):
        return self.event_names[self.i_event]

    def _current_landmark(self):
        return self.avail_landmarks[self.i_kp]

    def _rel_frames(self):
        return image_repo.get_rel_frames(
            self.subject_id,
            self.trial_id,
            self._current_event_name()
        )

    def _current_rel_frame(self):
        return self._rel_frames()[self.i_frame]

    def _current_cam_id(self):
        return self.cam_ids[self.i_cam]

    def _current_cam_ids(self):
        selected = self._current_cam_id()
        rest = [i for i in self.cam_ids if i != selected]
        return [selected] + rest

    def _get_image_paths(self):
        frame = image_repo.get_frame(self.subject_id, self.trial_id,
                                     self._current_event_name(),
                                     self._current_rel_frame()
                                     )
        cam_ids = self._current_cam_ids()
        return [frame.paths_by_cam[cam_id] for cam_id in cam_ids]

    def get_fig(self):
        return self.fig

    def onkeypress(self, event):
        if self.is_drawing:
            print("Busy ...")
            return

        # keypoint change
        if event.key == 'a':
            self.i_kp = (self.i_kp - 1) % len(self.avail_landmarks)
            print(" * Previous keypoint")
        elif event.key == 'd':
            self.i_kp = (self.i_kp + 1) % len(self.avail_landmarks)
            print(" * Next keypoint")
        elif event.key == 'z':
            self.i_event = (self.i_event - 1) % len(self.event_names)
            print(" * Previous event (%i)" % self.i_event)
        elif event.key == 'c':
            self.i_event = (self.i_event + 1) % len(self.event_names)
            print(" * Next event (%i)" % self.i_event)
        elif event.key == '/':
            self.i_cam = (self.i_cam - 1) % len(self.cam_ids)
        elif event.key == '*':
            self.i_cam = (self.i_cam + 1) % len(self.cam_ids)
        elif event.key == "+":
            n_relframes = len(self._rel_frames())
            self.i_frame = (self.i_frame + 1) % n_relframes
        elif event.key == "-":
            n_relframes = len(self._rel_frames())
            self.i_frame = (self.i_frame - 1) % n_relframes
        else:
            return

        self.is_drawing = True

        self.load_markers()
        self._draw_frame()
        self.is_drawing = False

    def _draw_initial_landmark_object(self):
        obj = self._current_landmark_object()

        if obj:
            return self.ax1.plot([obj["x"]], [obj["y"]], 'y+', markersize=10)[0]
        else:
            return self.ax1.plot([0, 0], 'y+', markersize=10)[0]

    def _draw_frame(self):
        obj = self._current_landmark_object()
        #print("_draw_frame", json.dumps(obj, indent=2))

        # render
        self.load_image(*self._get_image_paths())
        self.title.set_text(self._get_title())

        if obj:
            self.marker_line.set_data([obj["x"], obj["y"]])
            self.marker_line.set_visible(True)
        else:
            self.marker_line.set_visible(False)

        _cam_ids = self._current_cam_ids()
        self.ax1.set_xlabel(f"cam: {_cam_ids[0]}")
        self.ax2.set_xlabel(f"cam: {_cam_ids[1]}")

        # update canvas
        self.fig.canvas.flush_events()
        self.fig.draw_artist(self.marker_line)
        self.fig.draw_artist(self.title)
        self.fig.draw_artist(self.im1)
        self.fig.draw_artist(self.im2)
        plt.show()

    def _which_ax(self, event):
        for i, ax in enumerate(self.axes):
            if event.inaxes == ax:
                return i

    def onclick(self, event):
        if not event.inaxes:
            return

        if event.inaxes.figure.canvas.widgetlock.locked():
            return

        # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #      ('double' if event.dblclick else 'single', event.button,
        #       event.x, event.y, event.xdata, event.ydata))
        pane = self._which_ax(event)
        if event.button == 1 and pane == 0:
            self._set_marker(event.xdata, event.ydata)

    def match_lm(self, other):
        current_landmark = self.avail_landmarks[self.i_kp]
        current_cam_id = self._current_cam_id()

        cs = [
            other["cam_id"] == current_cam_id,
            other["landmark"] == current_landmark,
            other["event"] == self._current_event_name(),
        ]
        return all(cs)

    def _current_landmark_object(self):
        matched = list(filter(self.match_lm, self.markers))
        return matched[0] if matched else None

    def _set_marker(self, x, y):
        # get existing point id
        landmark = self._current_landmark_object()

        if landmark:
            # update
            print("   - Label (%s): subject='%s', trial=%i, event=%s, frame=%i, cam=%s, point=(%.2f, %.2f) update(id=%i)" % (
                landmark["landmark"],
                landmark["subject_id"],
                landmark["trial_id"],
                landmark["event"],
                landmark["relative_frame"],
                landmark["cam_id"],
                x, y,
                landmark["id"])
            )
            self.repo.update_point(landmark["id"], x, y)
        else:
            # insert
            print("   - Label (%s): subject='%s', trial=%i, event=%s, frame=%i, cam=%s, point=(%.2f, %.2f)" % (
                self._current_landmark(),
                self.subject_id,
                self.trial_id,
                self._current_event_name(),
                self._current_rel_frame(),
                self._current_cam_id(),
                x, y)
            )

            self.repo.create_point(self.subject_id,
                                   self.trial_id,
                                   self._current_event_name(),
                                   self._current_rel_frame(),
                                   self._current_cam_id(),
                                   self._current_landmark(),
                                   x, y
                                   )

        # progress frame
        self.next_point()

        # maintain strict consistency
        self.load_markers()

        # update view
        self._draw_frame()

    def _probe_frame(self):
        # 1) progress frame
        n_frames = len(self._rel_frames())
        i_frame = (self.i_frame + 1) % n_frames
        if i_frame > 0:
            print(".")
            return (i_frame, self.i_event, self.i_kp, self.i_cam)

        # 2) progress event
        n_events = len(self.event_names)
        i_event = (self.i_event + 1) % n_events
        if i_event > 0:
            print("..")
            return (i_frame, i_event, self.i_kp, self.i_cam)

        # 3) progress keypoint
        n_kps = len(self.avail_landmarks)
        i_kp = (self.i_kp + 1) % n_kps
        if i_kp > 0:
            print("...")
            return (i_frame, i_event, i_kp, self.i_cam)

        # 4) progress camera
        i_cam = (self.i_cam + 1) % len(self.cam_ids)
        return (i_frame, i_event, i_kp, i_cam)

    def has_frame(self, subject_id, trial_id, i_event, i_frame):
        # check if exists in image repo
        event_name = self.event_names[i_event]
        rel_frames = self.image_repo.get_rel_frames(
            subject_id,
            trial_id,
            event_name
        )
        rel_frame = rel_frames[i_frame]

        frame = self.image_repo.get_frame(subject_id,
                                          trial_id,
                                          event_name,
                                          rel_frame
                                          )
        return True if frame else False

    def next_point(self):
        limit = 1000
        i = 0
        while i < limit:
            i += 1

            # probe next
            i_frame, i_event, i_kp, i_cam = self._probe_frame()

            if self.has_frame(self.subject_id, self.trial_id, i_event, i_frame):
                break

        if i == limit:
            raise Exception("Next frame not found")

        print(" * [next point] frame found", i_event, self.event_names[i_event], i_frame)
        # update point...

        # 1) progress frame
        self.i_frame = i_frame
        self.i_event = i_event
        self.i_kp = i_kp
        self.i_cam = i_cam
        print("   - Progress")
        print("      - to frame:", self.i_frame)
        print("      - to event:", self.event_names[self.i_event], i_event)
        print("      - to landmark:", self.avail_landmarks[self.i_kp])
        print("      - to camera:", self.cam_ids[self.i_cam])

    def load_image(self, path1, path2):
        image1 = plt.imread(path1)
        image2 = plt.imread(path2)

        if self.im1 is None:
            self.im1 = self.ax1.imshow(image1)
        else:
            self.im1.set_array(image1)

        if self.im2 is None:
            self.im2 = self.ax2.imshow(image2)
        else:
            self.im2.set_array(image2)


parser = argparse.ArgumentParser()
parser.add_argument("input",
                    help="Path to input log .csv")
parser.add_argument("--db", "-D",
                    required=False,
                    default="test.db3",
                    help="Path to database file. Default: test.db3")
parser.add_argument("--subject", "-S",
                    required=True,
                    help="Subject id")
parser.add_argument("--trial", "-T",
                    type=int,
                    required=True,
                    help="Trial id")


if __name__ == "__main__":
    args = parser.parse_args()
    db_path = args.db
    frame_path = args.input

    print(" * Opening frame image log: %s" % (input,))
    image_repo = ImageRepo(frame_path)
    print(" * Opening label database: %s" % (db_path,))
    repo = SQLiteLabelRepo(db_path)

    print(" * Starting app...")
    app = LabellerApp(image_repo, repo, args.subject, args.trial)
    plt.show()
