from __future__ import annotations

import math

from PyQt6.QtCore import (
    QEasingCurve,
    QPointF,
    QPropertyAnimation,
    QRectF,
    QTimer,
    Qt,
    pyqtProperty,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor,
    QCursor,
    QPainter,
    QPen,
    QRadialGradient,
)
from PyQt6.QtWidgets import QWidget

from .design import Colors, Sizes


class BlueCursor(QWidget):
    """The little blue "buddy" dot that lives next to the user's cursor.

    Top-level frameless click-through window that:
      • continuously follows the system mouse cursor while tracking is on
        (polls QCursor.pos() at ~60fps and repositions the window in global
        screen coordinates),
      • can fly along a bezier arc to a target point when the LLM asks it to,
      • automatically returns to tracking the mouse a short delay after a fly,
      • renders as a lit blue sphere with an animated breathing glow.
    """

    arrived = pyqtSignal()

    TRACKING_INTERVAL_MS = 8  # ~120fps polling
    TRACKING_SMOOTHING = 0.10  # 0 = frozen, 1 = snap. lower = smoother, more lag.
    TRACKING_SNAP_THRESHOLD_PX = 0.25  # when within this distance, jump exactly onto target
    RESUME_TRACKING_DELAY_MS = 1500
    MOUSE_OFFSET = (4, 0)  # window offset from the OS cursor hotspot

    # Motion-stretch physics. The dot deforms along the line from itself toward
    # the mouse: a fast-moving cursor naturally pulls the lag-smoothed dot,
    # opening a gap that drives the stretch. When the cursor stops, the dot
    # catches up, the gap closes, and the stretch decays to zero on its own —
    # no explicit velocity tracking needed.
    STRETCH_MAX = 0.75           # peak amount; final scale becomes 1 + STRETCH_MAX along the pull
    STRETCH_DISTANCE_REFERENCE = 28.0  # gap (px) at which stretch saturates to STRETCH_MAX
    STRETCH_FOLLOW = 0.30        # how fast actual stretch chases the desired value
    STRETCH_PERP_SQUISH = 0.8    # how much the perpendicular axis squishes per unit stretch
    ANGLE_FOLLOW = 0.32          # how fast the stretch axis chases the pull direction
    ANGLE_MIN_GAP_PX = 1.0       # below this gap, leave the angle alone

    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.BypassWindowManagerHint
            | Qt.WindowType.WindowTransparentForInput,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        size = Sizes.CURSOR_SIZE
        self.resize(size, size)

        self._anim: QPropertyAnimation | None = None
        self._fly_progress = 0.0
        self._fly_start = QPointF(0, 0)
        self._fly_end = QPointF(0, 0)
        self._fly_control = QPointF(0, 0)

        self._tracking_timer = QTimer(self)
        self._tracking_timer.setInterval(self.TRACKING_INTERVAL_MS)
        self._tracking_timer.timeout.connect(self._on_tracking_tick)
        self._tracking = False
        # The current smoothed position in global coords as a float — keeping fractional
        # values across ticks is what prevents the integer-pixel rounding from killing
        # the smoothing at low velocities.
        self._tracked_pos: QPointF | None = None
        # Current rendered stretch magnitude (0 = round; STRETCH_MAX = max deform)
        # and the angle of the stretch axis in radians, both smoothed.
        self._stretch = 0.0
        self._stretch_angle = 0.0

        self._resume_timer = QTimer(self)
        self._resume_timer.setSingleShot(True)
        self._resume_timer.timeout.connect(self.start_tracking_mouse)

        # Continuous "breathing" pulse used by the paintEvent to modulate the
        # outer glow alpha. Loops forever; CPU cost is negligible because the
        # widget is tiny.
        self._pulse_phase = 0.0
        self._pulse_anim = QPropertyAnimation(self, b"pulsePhase")
        self._pulse_anim.setDuration(2400)
        self._pulse_anim.setStartValue(0.0)
        self._pulse_anim.setEndValue(1.0)
        self._pulse_anim.setLoopCount(-1)
        self._pulse_anim.setEasingCurve(QEasingCurve.Type.Linear)
        self._pulse_anim.start()

    # --- tracking -------------------------------------------------------------

    def start_tracking_mouse(self) -> None:
        """Begin smoothly following QCursor.pos() each tick. Idempotent."""
        self._resume_timer.stop()
        if self._tracking:
            return
        self._tracking = True
        # Snap to the target on the first frame so the user doesn't see the buddy
        # arc across the screen from wherever it was last.
        target = self._target_global()
        self._tracked_pos = target
        self._move_top_left_global(int(target.x()), int(target.y()))
        self.show()
        self.raise_()
        self._tracking_timer.start()

    def stop_tracking_mouse(self) -> None:
        self._tracking_timer.stop()
        self._tracking = False
        self._tracked_pos = None
        self._stretch = 0.0
        self._stretch_angle = 0.0

    def _target_global(self) -> QPointF:
        global_pos = QCursor.pos()
        return QPointF(
            global_pos.x() + self.MOUSE_OFFSET[0],
            global_pos.y() + self.MOUSE_OFFSET[1],
        )

    def _on_tracking_tick(self) -> None:
        target = self._target_global()
        if self._tracked_pos is None:
            self._tracked_pos = target

        # The gap between the actual cursor (where we WANT to be) and our
        # currently rendered position drives both the smoothing and the
        # squash-and-stretch — see _update_stretch_from_gap for the physics.
        gap_x = target.x() - self._tracked_pos.x()
        gap_y = target.y() - self._tracked_pos.y()

        if abs(gap_x) < self.TRACKING_SNAP_THRESHOLD_PX and abs(gap_y) < self.TRACKING_SNAP_THRESHOLD_PX:
            self._tracked_pos = target
        else:
            self._tracked_pos = QPointF(
                self._tracked_pos.x() + gap_x * self.TRACKING_SMOOTHING,
                self._tracked_pos.y() + gap_y * self.TRACKING_SMOOTHING,
            )
        self._move_top_left_global(int(self._tracked_pos.x()), int(self._tracked_pos.y()))
        self._update_stretch_from_gap(gap_x, gap_y)

    def _update_stretch_from_gap(self, gap_x: float, gap_y: float) -> None:
        """Drive squash-and-stretch from the lag distance to the cursor.

        Fast mouse motion opens a big gap → big stretch; the cursor stopping
        lets the dot catch up → gap closes → stretch decays. This is
        physically intuitive ("the dot stretches because it's being pulled")
        and decouples the deformation from the velocity sampling rate.
        """
        gap = math.hypot(gap_x, gap_y)
        desired_stretch = min(1.0, gap / self.STRETCH_DISTANCE_REFERENCE) * self.STRETCH_MAX
        self._stretch += (desired_stretch - self._stretch) * self.STRETCH_FOLLOW

        if gap > self.ANGLE_MIN_GAP_PX:
            desired_angle = math.atan2(gap_y, gap_x)
            delta = (desired_angle - self._stretch_angle + math.pi) % (2 * math.pi) - math.pi
            self._stretch_angle += delta * self.ANGLE_FOLLOW

        self.update()

    # --- pointing animation ---------------------------------------------------

    def fly_to(self, target_global_x: int, target_global_y: int, duration_ms: int = 700) -> None:
        """Animate from the current center to (target_global_x, target_global_y).

        Pauses mouse tracking for the duration of the animation; resumes tracking
        ``RESUME_TRACKING_DELAY_MS`` after arrival so the user has time to see
        where Clicky pointed before the triangle snaps back.
        """
        was_tracking = self._tracking
        self.stop_tracking_mouse()

        if self._anim is not None:
            self._anim.stop()

        start_center = self._current_center_global()
        end_center = QPointF(target_global_x, target_global_y)

        midpoint = QPointF(
            (start_center.x() + end_center.x()) / 2,
            (start_center.y() + end_center.y()) / 2,
        )
        distance = (
            (end_center.x() - start_center.x()) ** 2
            + (end_center.y() - start_center.y()) ** 2
        ) ** 0.5
        arc_height = min(distance * 0.25, 180)
        control = QPointF(midpoint.x(), midpoint.y() - arc_height)

        self._fly_start = start_center
        self._fly_end = end_center
        self._fly_control = control

        self.show()
        self.raise_()

        animation = QPropertyAnimation(self, b"flyProgress")
        animation.setDuration(duration_ms)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.finished.connect(self.arrived.emit)
        if was_tracking:
            animation.finished.connect(
                lambda: self._resume_timer.start(self.RESUME_TRACKING_DELAY_MS)
            )
        animation.start()
        self._anim = animation

    def show_at(self, global_x: int, global_y: int) -> None:
        """Place the cursor center at the given global point, no animation."""
        self._move_center_to_global(QPointF(global_x, global_y))
        self.show()
        self.raise_()

    # --- coordinate helpers (top-level window uses global coords directly) ---

    def _current_center_global(self) -> QPointF:
        return QPointF(self.x() + self.width() / 2, self.y() + self.height() / 2)

    def _move_center_to_global(self, center: QPointF) -> None:
        self.move(int(center.x() - self.width() / 2), int(center.y() - self.height() / 2))

    def _move_top_left_global(self, gx: int, gy: int) -> None:
        self.move(gx, gy)

    # --- animation property --------------------------------------------------

    def get_fly_progress(self) -> float:
        return self._fly_progress

    def set_fly_progress(self, value: float) -> None:
        self._fly_progress = value
        t = max(0.0, min(1.0, value))
        one_minus_t = 1.0 - t
        point = QPointF(
            one_minus_t * one_minus_t * self._fly_start.x()
            + 2 * one_minus_t * t * self._fly_control.x()
            + t * t * self._fly_end.x(),
            one_minus_t * one_minus_t * self._fly_start.y()
            + 2 * one_minus_t * t * self._fly_control.y()
            + t * t * self._fly_end.y(),
        )
        self._move_center_to_global(point)

    flyProgress = pyqtProperty(float, fget=get_fly_progress, fset=set_fly_progress)

    # --- pulse property -----------------------------------------------------

    def get_pulse_phase(self) -> float:
        return self._pulse_phase

    def set_pulse_phase(self, value: float) -> None:
        self._pulse_phase = value
        self.update()

    pulsePhase = pyqtProperty(float, fget=get_pulse_phase, fset=set_pulse_phase)

    # --- painting -----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: ARG002 — Qt signature
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        cx = w / 2
        cy = h / 2

        # Half-sine pulse in [0, 1] driven by phase ∈ [0, 1).
        pulse = 0.5 - 0.5 * math.cos(self._pulse_phase * math.pi * 2.0)

        # --- outer breathing glow (drawn outside the stretch transform so it
        # reads as a soft ambient halo that doesn't physically deform with the
        # body, and also so the glow never gets clipped at the widget edges
        # when the body stretches hard) -------------------------------------
        outer_radius = w * 0.26
        outer = QRadialGradient(cx, cy, outer_radius)
        glow_color = QColor(Colors.CURSOR_HALO)
        glow_color.setAlpha(int(60 + 50 * pulse))
        outer.setColorAt(0.0, glow_color)
        mid_color = QColor(glow_color)
        mid_color.setAlpha(int(28 + 24 * pulse))
        outer.setColorAt(0.55, mid_color)
        outer.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(outer)
        painter.drawEllipse(QRectF(cx - outer_radius, cy - outer_radius, outer_radius * 2, outer_radius * 2))

        # --- squash-and-stretch transform for the rigid sphere bits --------
        painter.save()
        if self._stretch > 0.005:
            painter.translate(cx, cy)
            painter.rotate(math.degrees(self._stretch_angle))
            sx = 1.0 + self._stretch
            sy = 1.0 / (1.0 + self._stretch * self.STRETCH_PERP_SQUISH)
            painter.scale(sx, sy)
            painter.translate(-cx, -cy)

        # --- sphere body ---------------------------------------------------
        # Off-center radial gradient so the highlight sits at the upper-left
        # and the body deepens toward the lower-right — that's what reads as
        # "lit sphere".
        body_radius = w * 0.15
        hl_x = cx - body_radius * 0.35
        hl_y = cy - body_radius * 0.35
        body = QRadialGradient(hl_x, hl_y, body_radius * 1.7)
        body.setColorAt(0.0, QColor(Colors.CURSOR_HIGHLIGHT))
        body.setColorAt(0.18, QColor(Colors.CURSOR_LIGHT))
        body.setColorAt(0.55, QColor(Colors.CURSOR_MID))
        body.setColorAt(1.0, QColor(Colors.CURSOR_DEEP))
        painter.setBrush(body)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), body_radius, body_radius)

        # --- crisp specular highlight (small white blob, slightly stretched) ---
        spec_rx = body_radius * 0.42
        spec_ry = body_radius * 0.30
        spec_cx = cx - body_radius * 0.32
        spec_cy = cy - body_radius * 0.42
        spec = QRadialGradient(spec_cx, spec_cy, spec_rx)
        white = QColor(255, 255, 255, 230)
        spec.setColorAt(0.0, white)
        soft_white = QColor(255, 255, 255, 0)
        spec.setColorAt(1.0, soft_white)
        painter.setBrush(spec)
        painter.drawEllipse(QPointF(spec_cx, spec_cy), spec_rx, spec_ry)

        # --- subtle dark rim for definition --------------------------------
        rim = QColor(Colors.CURSOR_RIM)
        rim.setAlpha(170)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(rim, 0.9))
        painter.drawEllipse(QPointF(cx, cy), body_radius, body_radius)

        painter.restore()
