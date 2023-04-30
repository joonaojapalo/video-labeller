from matplotlib.backend_tools import Cursors


class BlittedCursor:
    """
    A cross-hair cursor using blitting for faster redraw.
    """

    def __init__(self, ax, color='white'):
        self.ax = ax
        self.background = None
        self.horizontal_line = ax.plot(
            [], [], color=color, lw=0.8, ls='-', alpha=0.5)[0]
        self.vertical_line = ax.plot(
            [], [], color=color, lw=0.8, ls='-', alpha=0.5)[0]

        # text location in axes coordinates
        self.text = ax.text(0.72, 0.9, '', transform=ax.transAxes)
        self._creating_background = False
        ax.figure.canvas.mpl_connect('draw_event', self.on_draw)

    def on_draw(self, event):
        self.create_new_background()

    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        self.text.set_visible(visible)
        return need_redraw

    def create_new_background(self):
        if self._creating_background:
            # discard calls triggered from within this function
            return
        self._creating_background = True
        self.set_cross_hair_visible(False)
        self.ax.figure.canvas.draw()
        self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)
        self.set_cross_hair_visible(True)
        self._creating_background = False

    def on_mouse_move(self, event):
        if self.ax.figure.canvas.widgetlock.locked():
            return

        if self.background is None:
            self.create_new_background()

        if not event.inaxes or event.inaxes != self.ax:
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax.figure.canvas.restore_region(self.background)
                self.ax.figure.canvas.blit(self.ax.bbox)
                self.ax.figure.canvas.set_cursor(Cursors.SELECT_REGION)
        else:
            # hide cursor
            self.ax.figure.canvas._tkcanvas.configure(cursor="none")
            self.set_cross_hair_visible(True)

            # zoom invariant size
            x0, x1 = self.ax.get_xlim()
            res = int(self.ax.figure.get_figwidth() * self.ax.figure.dpi)
            d = (x1-x0) / res * 9

            # update the line positions
            x, y = event.xdata, event.ydata
            self.horizontal_line.set_data([x-d, x+d], [y, y])
            self.vertical_line.set_data([x, x], [y-d, y+d])
            self.text.set_text('x=%1.2f, y=%1.2f' % (x, y))

            self.ax.figure.canvas.restore_region(self.background)
            self.ax.draw_artist(self.horizontal_line)
            self.ax.draw_artist(self.vertical_line)
            self.ax.draw_artist(self.text)
            self.ax.figure.canvas.blit(self.ax.bbox)
