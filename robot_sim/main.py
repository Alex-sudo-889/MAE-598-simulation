from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import controls
import numpy as np
import pygame as pg
from sim import Simulator, load_default_scene, load_scene

WIDTH, HEIGHT = 1380, 760
BACKGROUND = (239, 240, 243)
PANEL_WIDTH = 420
PANEL_BG = (250, 250, 250)
PANEL_TEXT = (30, 30, 30)
PANEL_ACCENT = (110, 110, 110)
SLIDER_TRACK = (210, 210, 210)
SLIDER_HANDLE = (70, 130, 220)
CHECKBOX_BG = (255, 255, 255)
CHECKBOX_ACTIVE = (60, 160, 80)
TAB_INACTIVE = (220, 220, 220)
TAB_ACTIVE = (200, 205, 215)
BUTTON_BG = (225, 230, 245)
BUTTON_BORDER = (110, 140, 210)
BUTTON_TEXT = (25, 25, 25)
MESSAGE_ERROR = (200, 70, 70)
MESSAGE_SUCCESS = (50, 150, 90)
EDITOR_BORDER = (200, 200, 200)


def _format_constants(constants: Dict[str, object]) -> str:
    if not constants:
        return ""
    parts = []
    for key, value in constants.items():
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def _parse_constants(text: str) -> Dict[str, float]:
    constants: Dict[str, float] = {}
    if not text:
        return constants
    raw = text.replace(";", ",").split(",")
    for token in raw:
        token = token.strip()
        if not token or "=" not in token:
            continue
        name, value = token.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            continue
        try:
            constants[name] = float(value)
        except ValueError:
            constants[name] = value
    return constants


class InputField:
    def __init__(self, text: str = "", width: int = 160) -> None:
        self.text = text
        self.cursor = len(text)
        self.active = False
        self.rect = pg.Rect(0, 0, width, 24)
        self.width = width

    def set_rect(self, rect: pg.Rect) -> None:
        self.rect = rect
        self.width = rect.width

    def handle_mouse(self, event: pg.event.Event, font: pg.font.Font) -> bool:
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            was_active = self.active
            self.active = self.rect.collidepoint(event.pos)
            if self.active:
                rel_x = event.pos[0] - self.rect.x
                self.cursor = self._cursor_from_mouse(rel_x, font)
            elif was_active:
                self.cursor = len(self.text)
            return self.active != was_active
        return False

    def handle_key(self, event: pg.event.Event) -> bool:
        if not self.active or event.type != pg.KEYDOWN:
            return False
        handled = True
        if event.key in (pg.K_RETURN, pg.K_KP_ENTER):
            self.active = False
            self.cursor = min(self.cursor, len(self.text))
        elif event.key == pg.K_BACKSPACE:
            if self.cursor > 0:
                self.text = self.text[: self.cursor - 1] + self.text[self.cursor :]
                self.cursor -= 1
        elif event.key == pg.K_DELETE:
            if self.cursor < len(self.text):
                self.text = self.text[: self.cursor] + self.text[self.cursor + 1 :]
        elif event.key == pg.K_LEFT:
            self.cursor = max(0, self.cursor - 1)
        elif event.key == pg.K_RIGHT:
            self.cursor = min(len(self.text), self.cursor + 1)
        elif event.key == pg.K_HOME:
            self.cursor = 0
        elif event.key == pg.K_END:
            self.cursor = len(self.text)
        else:
            if event.unicode and event.unicode.isprintable():
                self.text = (
                    self.text[: self.cursor] + event.unicode + self.text[self.cursor :]
                )
                self.cursor += len(event.unicode)
        return handled

    def draw(self, surface: pg.Surface, font: pg.font.Font) -> None:
        pg.draw.rect(surface, CHECKBOX_BG, self.rect, border_radius=4)
        pg.draw.rect(surface, EDITOR_BORDER, self.rect, 1, border_radius=4)
        text_surf = font.render(self.text, True, PANEL_TEXT)
        surface.blit(text_surf, (self.rect.x + 4, self.rect.y + 4))
        if self.active:
            prefix = font.render(self.text[: self.cursor], True, PANEL_TEXT)
            caret_x = self.rect.x + 4 + prefix.get_width()
            caret_y = self.rect.y + 4
            pg.draw.line(
                surface,
                PANEL_TEXT,
                (caret_x, caret_y),
                (caret_x, caret_y + font.get_height() - 4),
                1,
            )

    def _cursor_from_mouse(self, rel_x: float, font: pg.font.Font) -> int:
        pos = 0
        for idx, ch in enumerate(self.text):
            ch_width = font.size(ch)[0]
            if rel_x < pos + ch_width * 0.6:
                return idx
            pos += ch_width
        return len(self.text)


class TermBox:
    def __init__(self, info, code_font: pg.font.Font) -> None:
        self.name = info.name
        self.label = info.label or info.name
        self.description = info.description
        self.kind = info.kind
        self.weight = info.weight
        self.active = info.active
        self.fields: Dict[str, InputField] = {}
        self.code_font = code_font
        if self.kind != "potential":
            self.fields["vx"] = InputField(info.vx_expr, width=200)
            self.fields["vy"] = InputField(info.vy_expr, width=200)
        else:
            self.fields["potential"] = InputField(info.potential_expr, width=200)
        if self.kind in ("obstacle", "robots"):
            self.fields["condition"] = InputField(info.condition_expr, width=200)
        self.fields["constants"] = InputField(
            _format_constants(info.constants), width=200
        )
        self.original_values = self._snapshot_fields()
        self.rect = pg.Rect(0, 0, 0, 0)

    def _snapshot_fields(self) -> Dict[str, str]:
        return {name: field.text for name, field in self.fields.items()}

    def is_dirty(self) -> bool:
        current = {name: field.text for name, field in self.fields.items()}
        return current != self.original_values

    def handle_event(self, event: pg.event.Event) -> bool:
        handled = False
        if event.type == pg.MOUSEBUTTONDOWN:
            for field in self.fields.values():
                handled = field.handle_mouse(event, self.code_font) or handled
        elif event.type == pg.KEYDOWN:
            for field in self.fields.values():
                handled = field.handle_key(event) or handled
        return handled

    def draw(
        self,
        surface: pg.Surface,
        x: int,
        y: int,
        width: int,
        label_font: pg.font.Font,
        text_font: pg.font.Font,
    ) -> int:
        padding = 10
        box_width = width - 16
        title_height = label_font.get_height()
        status_height = text_font.get_height() + 4
        desc_height = text_font.get_height() + 6 if self.description else 0
        field_specs: List[Tuple[str, str]] = []
        if self.kind == "potential":
            field_specs.append(("potential", "potential ="))
        else:
            field_specs.append(("vx", "vx ="))
            field_specs.append(("vy", "vy ="))
            if self.kind in ("obstacle", "robots"):
                field_specs.append(("condition", "condition ="))
        field_specs.append(("constants", "constants ="))
        field_height = 26
        total_field_height = len(field_specs) * (field_height + 6)
        height = (
            padding * 2
            + title_height
            + status_height
            + desc_height
            + total_field_height
        )
        box_rect = pg.Rect(x, y, box_width, height)
        self.rect = box_rect
        border_color = MESSAGE_ERROR if self.is_dirty() else BUTTON_BORDER
        pg.draw.rect(surface, BUTTON_BG, box_rect, border_radius=8)
        pg.draw.rect(surface, border_color, box_rect, 1, border_radius=8)
        cursor = y + padding
        title = label_font.render(f"{self.label} ({self.name})", True, PANEL_TEXT)
        surface.blit(title, (x + padding, cursor))
        cursor += title.get_height() + 2
        status = f"{self.kind} | {'ON' if self.active else 'OFF'} | w={self.weight:.2f}"
        status_surf = text_font.render(status, True, PANEL_ACCENT)
        surface.blit(status_surf, (x + padding, cursor))
        cursor += status_surf.get_height() + 2
        if self.description:
            desc = text_font.render(self.description, True, PANEL_ACCENT)
            surface.blit(desc, (x + padding, cursor))
            cursor += desc.get_height() + 4
        label_width = 96
        for key, label in field_specs:
            field = self.fields.get(key)
            if not field:
                continue
            label_surface = text_font.render(label, True, PANEL_ACCENT)
            surface.blit(label_surface, (x + padding, cursor + 4))
            input_rect = pg.Rect(
                x + padding + label_width,
                cursor,
                box_width - label_width - padding,
                field_height,
            )
            field.set_rect(input_rect)
            field.draw(surface, self.code_font)
            cursor += field_height + 6
        return height + 12

    def to_entry(self) -> Dict[str, object]:
        entry = {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "weight": self.weight,
            "active": self.active,
            "kind": self.kind,
            "vx": self.fields.get("vx").text if "vx" in self.fields else "0.0",
            "vy": self.fields.get("vy").text if "vy" in self.fields else "0.0",
            "condition": self.fields.get("condition").text
            if "condition" in self.fields
            else "",
            "potential": self.fields.get("potential").text
            if "potential" in self.fields
            else "",
            "constants": _parse_constants(
                self.fields.get("constants", InputField("")).text
            ),
        }
        return entry

    def reset_to_original(self) -> None:
        for name, value in self.original_values.items():
            self.fields[name].text = value
            self.fields[name].cursor = len(value)
        self.mark_clean()

    def mark_clean(self) -> None:
        self.original_values = self._snapshot_fields()


ROOT = Path(__file__).resolve().parent
SCENE_FILE = ROOT / "scene.json"
SNAPSHOT_DIR = ROOT / "snapshots"
LOG_DIR = ROOT / "logs"

TERM_KEYS = {
    pg.K_1: "go_to",
    pg.K_2: "avoid_obs",
    pg.K_3: "avoid_robots",
    pg.K_4: "bounds",
    pg.K_5: "phi",
}


class Overlay:
    def __init__(self, rect: pg.Rect, font: pg.font.Font, sim: Simulator) -> None:
        self.rect = rect
        self.font = font
        self.small_font = pg.font.Font(None, 18)
        code_font_name = pg.font.match_font("consolas,menlo,courier,monospace")
        self.code_font = pg.font.Font(code_font_name or None, 16)
        self.term_title_font = pg.font.Font(None, 22)
        self.term_body_font = pg.font.Font(None, 18)
        self.sim = sim
        self.slider_active: Optional[str] = None
        self.slider_rects: Dict[str, pg.Rect] = {}
        self.toggle_rects: Dict[str, pg.Rect] = {}
        self.tab_rects: Dict[str, pg.Rect] = {}
        self.save_rect: Optional[pg.Rect] = None
        self.revert_rect: Optional[pg.Rect] = None
        self.json_reload_rect: Optional[pg.Rect] = None
        self.count_button_rects: Dict[str, Dict[str, pg.Rect]] = {
            "robots": {},
            "targets": {},
        }
        self.mode: str = "control"
        self.message_text: Optional[str] = None
        self.message_color = PANEL_ACCENT
        self.message_until = 0
        self.term_boxes: List[TermBox] = []
        self.config_dirty = False
        self.control_scroll = 0.0
        self.config_scroll = 0.0
        self.control_content_height = 0.0
        self.config_content_height = 0.0
        self._build_term_boxes()

    def set_mode(self, mode: str) -> None:
        if mode not in {"control", "config"}:
            return
        if self.mode != mode:
            self.mode = mode
            self.slider_active = None

    def toggle_mode(self) -> None:
        self.set_mode("config" if self.mode == "control" else "control")

    def set_simulator(self, sim: Simulator) -> None:
        self.sim = sim

    def notify(self, text: str, color: Tuple[int, int, int]) -> None:
        self.message_text = text
        self.message_color = color
        self.message_until = pg.time.get_ticks() + 2200

    def draw(
        self, surface: pg.Surface, fps: float, paused: bool, selection_text: str
    ) -> None:
        pg.draw.rect(surface, PANEL_BG, self.rect)
        self._draw_tabs(surface)
        if self.mode == "control":
            self._draw_control_panel(surface, fps, paused, selection_text)
        else:
            self._draw_config_panel(surface)
        self._draw_message(surface)

    def handle_event(self, event: pg.event.Event) -> None:
        if (
            event.type == pg.MOUSEBUTTONDOWN
            and event.button == 1
            and self._handle_tab_click(event.pos)
        ):
            return
        if event.type == pg.MOUSEWHEEL and self.rect.collidepoint(pg.mouse.get_pos()):
            self._scroll_panel(self.mode, -event.y * 30.0)
            return
        if self.mode == "control":
            self._handle_control_events(event)
        else:
            self._handle_config_events(event)
            if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                handled = False
                for box in self.term_boxes:
                    handled = box.handle_event(event) or handled
                if handled:
                    self._update_dirty_flag()

    def handle_keydown(self, event: pg.event.Event) -> bool:
        ctrl = event.mod & pg.KMOD_CTRL
        if self.mode == "config":
            if event.key == pg.K_F5 or (event.key == pg.K_s and ctrl):
                self._save_term_boxes()
                return True
            if event.key == pg.K_r and ctrl:
                self._build_term_boxes()
                self.notify("Discarded term edits", PANEL_ACCENT)
                return True
            handled = False
            for box in self.term_boxes:
                handled = box.handle_event(event) or handled
            if handled:
                self._update_dirty_flag()
                return True
        return False

    def _handle_control_events(self, event: pg.event.Event) -> None:
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                for name, rect in self.toggle_rects.items():
                    if rect.collidepoint(event.pos):
                        controls.toggle_term(name)
                        return
                for name, rect in self.slider_rects.items():
                    if rect.collidepoint(event.pos):
                        self.slider_active = name
                        self._update_slider(name, event.pos[0])
                        return
        elif event.type == pg.MOUSEBUTTONUP and event.button == 1:
            self.slider_active = None
        elif event.type == pg.MOUSEMOTION and self.slider_active:
            self._update_slider(self.slider_active, event.pos[0])

    def _handle_config_events(self, event: pg.event.Event) -> None:
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            if self.save_rect and self.save_rect.collidepoint(event.pos):
                self._save_term_boxes()
                return
            if self.revert_rect and self.revert_rect.collidepoint(event.pos):
                self._build_term_boxes()
                self.notify("Discarded term edits", PANEL_ACCENT)
                return
            if self.json_reload_rect and self.json_reload_rect.collidepoint(event.pos):
                success, message = controls.reload_json_terms()
                color = MESSAGE_SUCCESS if success else MESSAGE_ERROR
                self.notify(message, color)
                if success:
                    self._build_term_boxes()
                return
            for kind, rects in self.count_button_rects.items():
                for action, rect in rects.items():
                    if rect.collidepoint(event.pos):
                        self._apply_count_action(kind, action)
                        return

    def _handle_tab_click(self, pos: Tuple[int, int]) -> bool:
        for mode, rect in self.tab_rects.items():
            if rect.collidepoint(pos):
                self.set_mode(mode)
                return True
        return False

    def _draw_tabs(self, surface: pg.Surface) -> None:
        tab_height = 32
        control_rect = pg.Rect(
            self.rect.x, self.rect.y, self.rect.width // 2, tab_height
        )
        config_rect = pg.Rect(
            control_rect.right,
            self.rect.y,
            self.rect.width - control_rect.width,
            tab_height,
        )
        self.tab_rects = {"control": control_rect, "config": config_rect}
        for mode, rect in self.tab_rects.items():
            color = TAB_ACTIVE if mode == self.mode else TAB_INACTIVE
            pg.draw.rect(surface, color, rect)
            label = "Controls" if mode == "control" else "Config"
            text = self.small_font.render(label, True, PANEL_TEXT)
            text_pos = (rect.centerx - text.get_width() // 2, rect.y + 7)
            surface.blit(text, text_pos)

    def _draw_control_panel(
        self, surface: pg.Surface, fps: float, paused: bool, selection_text: str
    ) -> None:
        self.slider_rects.clear()
        self.toggle_rects.clear()
        self.save_rect = None
        self.json_reload_rect = None
        self.count_button_rects = {"robots": {}, "targets": {}}
        content_top = self.rect.y + 32
        scroll = self.control_scroll
        cursor = 8.0
        header = self.font.render("Robot Sim", True, PANEL_TEXT)
        surface.blit(header, (self.rect.x + 16, content_top + cursor - scroll))
        cursor += header.get_height() + 6
        status = f"FPS: {fps:4.1f}  {'PAUSED' if paused else 'RUN'}"
        status_surf = self.small_font.render(status, True, PANEL_ACCENT)
        surface.blit(status_surf, (self.rect.x + 16, content_top + cursor - scroll))
        cursor += status_surf.get_height() + 4
        if selection_text:
            select_surf = self.small_font.render(
                f"Selected: {selection_text}", True, PANEL_ACCENT
            )
            surface.blit(select_surf, (self.rect.x + 16, content_top + cursor - scroll))
            cursor += select_surf.get_height() + 6
        else:
            cursor += 6
        for name, info in controls.iter_terms():
            toggle_rect = pg.Rect(
                self.rect.x + 18, content_top + cursor - scroll, 16, 16
            )
            pg.draw.rect(surface, CHECKBOX_BG, toggle_rect, border_radius=3)
            if info.active:
                pg.draw.rect(surface, CHECKBOX_ACTIVE, toggle_rect.inflate(-4, -4))
            label = self.small_font.render(info.label or name, True, PANEL_TEXT)
            surface.blit(
                label, (toggle_rect.right + 8, content_top + cursor - scroll - 2)
            )
            slider_y = content_top + cursor - scroll + 28
            slider_rect = pg.Rect(self.rect.x + 18, slider_y, self.rect.width - 36, 8)
            pg.draw.rect(surface, SLIDER_TRACK, slider_rect, border_radius=4)
            knob_x = slider_rect.x + int((info.weight / 2.0) * slider_rect.width)
            knob_rect = pg.Rect(knob_x - 6, slider_rect.y - 4, 12, 16)
            pg.draw.rect(surface, SLIDER_HANDLE, knob_rect, border_radius=4)
            value_surf = self.small_font.render(
                f"{info.weight:.2f}", True, PANEL_ACCENT
            )
            surface.blit(value_surf, (slider_rect.x, slider_rect.y + 12))
            desc_surf = self.small_font.render(info.description, True, PANEL_ACCENT)
            surface.blit(desc_surf, (slider_rect.x, slider_rect.y + 28))
            self.slider_rects[name] = slider_rect
            self.toggle_rects[name] = toggle_rect
            cursor += 72
        hint_lines = [
            "Mouse wheel: scroll panel",
            "Space: pause",
            "R: reset  Ctrl+S: save scene",
            "S: snapshot  L: log CSV",
            "A/G/O: add robot/target/obstacle",
            "D/Delete: remove selection",
            "Ctrl+Tab: open math config",
            "Config tab: edit math + F5/Ctrl+S to save JSON",
        ]
        cursor += 8
        for line in hint_lines:
            surf = self.small_font.render(line, True, PANEL_ACCENT)
            surface.blit(surf, (self.rect.x + 16, content_top + cursor - scroll))
            cursor += 18
        self._update_scroll_limit("control", cursor + 16)

    def _draw_config_panel(self, surface: pg.Surface) -> None:
        info = controls.get_config_state()
        self.slider_rects.clear()
        self.toggle_rects.clear()
        self.save_rect = None
        self.revert_rect = None
        self.json_reload_rect = None
        self.count_button_rects = {"robots": {}, "targets": {}}
        content_top = self.rect.y + 32
        scroll = self.config_scroll
        cursor = 8.0
        heading = self.font.render("Control Term Config", True, PANEL_TEXT)
        surface.blit(heading, (self.rect.x + 16, content_top + cursor - scroll))
        cursor += heading.get_height() + 6
        json_path = info.get("json_path")
        json_name = json_path.name if json_path else "term_config.json"
        json_label = self.small_font.render(f"JSON file: {json_name}", True, PANEL_TEXT)
        surface.blit(json_label, (self.rect.x + 16, content_top + cursor - scroll))
        cursor += json_label.get_height() + 2
        status = info.get("status") or "Ready"
        status_color = PANEL_ACCENT if not info.get("error") else MESSAGE_ERROR
        status_surf = self.small_font.render(f"Runtime: {status}", True, status_color)
        surface.blit(status_surf, (self.rect.x + 16, content_top + cursor - scroll))
        cursor += status_surf.get_height() + 2
        dirty_text = (
            "Pending edits (not saved)" if self.config_dirty else "All edits saved"
        )
        dirty_color = MESSAGE_ERROR if self.config_dirty else PANEL_ACCENT
        dirty_surf = self.small_font.render(dirty_text, True, dirty_color)
        surface.blit(dirty_surf, (self.rect.x + 16, content_top + cursor - scroll))
        cursor += dirty_surf.get_height() + 4
        last_saved = info.get("last_json_saved")
        saved_text = (
            f"JSON saved: {last_saved.strftime('%H:%M:%S')}"
            if last_saved
            else "JSON saved: --"
        )
        saved_surf = self.small_font.render(saved_text, True, PANEL_ACCENT)
        surface.blit(saved_surf, (self.rect.x + 16, content_top + cursor - scroll))
        cursor += saved_surf.get_height() + 2
        last_loaded = info.get("last_json_loaded")
        loaded_text = (
            f"JSON applied: {last_loaded.strftime('%H:%M:%S')}"
            if last_loaded
            else "JSON applied: --"
        )
        loaded_surf = self.small_font.render(loaded_text, True, PANEL_ACCENT)
        surface.blit(loaded_surf, (self.rect.x + 16, content_top + cursor - scroll))
        cursor += loaded_surf.get_height() + 8
        btn_width = self.rect.width - 36
        save_rect = pg.Rect(
            self.rect.x + 18, content_top + cursor - scroll, btn_width, 32
        )
        self.save_rect = save_rect
        pg.draw.rect(surface, BUTTON_BG, save_rect, border_radius=6)
        pg.draw.rect(surface, BUTTON_BORDER, save_rect, 2, border_radius=6)
        save_label = self.small_font.render(
            "Save JSON (F5 / Ctrl+S)", True, BUTTON_TEXT
        )
        surface.blit(
            save_label,
            (
                save_rect.centerx - save_label.get_width() // 2,
                save_rect.centery - save_label.get_height() // 2,
            ),
        )
        cursor += save_rect.height + 10
        revert_rect = pg.Rect(
            self.rect.x + 18, content_top + cursor - scroll, btn_width, 28
        )
        self.revert_rect = revert_rect
        pg.draw.rect(surface, BUTTON_BG, revert_rect, border_radius=6)
        pg.draw.rect(surface, BUTTON_BORDER, revert_rect, 1, border_radius=6)
        revert_label = self.small_font.render(
            "Discard edits (Ctrl+R)", True, BUTTON_TEXT
        )
        surface.blit(
            revert_label,
            (
                revert_rect.centerx - revert_label.get_width() // 2,
                revert_rect.centery - revert_label.get_height() // 2,
            ),
        )
        cursor += revert_rect.height + 8
        json_rect = pg.Rect(
            self.rect.x + 18, content_top + cursor - scroll, btn_width, 28
        )
        self.json_reload_rect = json_rect
        pg.draw.rect(surface, BUTTON_BG, json_rect, border_radius=6)
        pg.draw.rect(surface, BUTTON_BORDER, json_rect, 1, border_radius=6)
        json_reload_label = self.small_font.render(
            "Reload JSON + apply", True, BUTTON_TEXT
        )
        surface.blit(
            json_reload_label,
            (
                json_rect.centerx - json_reload_label.get_width() // 2,
                json_rect.centery - json_reload_label.get_height() // 2,
            ),
        )
        cursor += json_rect.height + 12
        cursor = self._draw_count_controls(
            surface, "robots", len(self.sim.robots), cursor, scroll, content_top
        )
        cursor = self._draw_count_controls(
            surface, "targets", len(self.sim.targets), cursor, scroll, content_top
        )
        cursor += 4
        section = self.small_font.render(
            "Edit control terms (pure math):", True, PANEL_TEXT
        )
        surface.blit(section, (self.rect.x + 16, content_top + cursor - scroll))
        cursor += section.get_height() + 4
        panel_width = self.rect.width - 32
        x = self.rect.x + 16
        if not self.term_boxes:
            none = self.small_font.render("(no terms configured)", True, PANEL_ACCENT)
            surface.blit(none, (x, content_top + cursor - scroll))
            cursor += none.get_height() + 8
        else:
            for box in self.term_boxes:
                height = box.draw(
                    surface,
                    x,
                    content_top + cursor - scroll,
                    panel_width,
                    self.term_title_font,
                    self.term_body_font,
                )
                cursor += height
        cursor += 6
        info_lines = [
            "vx/vy apply per robot; obstacle/robot terms use 'condition'.",
            "potential defines phi(x,y) and engine applies the negative gradient.",
            "constants = comma list (k=200, infl=80). Leave blank to clear.",
        ]
        for line in info_lines:
            tip = self.small_font.render(line, True, PANEL_ACCENT)
            surface.blit(tip, (self.rect.x + 16, content_top + cursor - scroll))
            cursor += tip.get_height() + 2
        cursor += 6
        self._update_scroll_limit("config", cursor)

    def _draw_count_controls(
        self,
        surface: pg.Surface,
        kind: str,
        count: int,
        cursor: float,
        scroll: float,
        content_top: float,
    ) -> float:
        label = f"{kind.capitalize()} count: {count}"
        label_surf = self.small_font.render(label, True, PANEL_TEXT)
        surface.blit(label_surf, (self.rect.x + 16, content_top + cursor - scroll))
        cursor += label_surf.get_height() + 4
        width = 32
        spacing = 6
        btn_y = content_top + cursor - scroll
        minus_rect = pg.Rect(self.rect.x + 18, btn_y, width, 24)
        zero_rect = pg.Rect(minus_rect.right + spacing, btn_y, width, 24)
        plus_rect = pg.Rect(zero_rect.right + spacing, btn_y, width, 24)
        pg.draw.rect(surface, BUTTON_BG, minus_rect, border_radius=4)
        pg.draw.rect(surface, BUTTON_BG, zero_rect, border_radius=4)
        pg.draw.rect(surface, BUTTON_BG, plus_rect, border_radius=4)
        pg.draw.rect(surface, BUTTON_BORDER, minus_rect, 1, border_radius=4)
        pg.draw.rect(surface, BUTTON_BORDER, zero_rect, 1, border_radius=4)
        pg.draw.rect(surface, BUTTON_BORDER, plus_rect, 1, border_radius=4)
        minus_label = self.small_font.render("-", True, BUTTON_TEXT)
        zero_label = self.small_font.render("0", True, BUTTON_TEXT)
        plus_label = self.small_font.render("+", True, BUTTON_TEXT)
        surface.blit(
            minus_label,
            (
                minus_rect.centerx - minus_label.get_width() // 2,
                minus_rect.centery - minus_label.get_height() // 2,
            ),
        )
        surface.blit(
            zero_label,
            (
                zero_rect.centerx - zero_label.get_width() // 2,
                zero_rect.centery - zero_label.get_height() // 2,
            ),
        )
        surface.blit(
            plus_label,
            (
                plus_rect.centerx - plus_label.get_width() // 2,
                plus_rect.centery - plus_label.get_height() // 2,
            ),
        )
        self.count_button_rects[kind] = {
            "dec": minus_rect,
            "zero": zero_rect,
            "inc": plus_rect,
        }
        cursor += 24 + 10
        return cursor

    def _update_scroll_limit(self, mode: str, content_height: float) -> None:
        viewport = max(1.0, self.rect.height - 32)
        max_scroll = max(0.0, content_height - viewport)
        if mode == "control":
            self.control_content_height = content_height
            self.control_scroll = max(0.0, min(self.control_scroll, max_scroll))
        else:
            self.config_content_height = content_height
            self.config_scroll = max(0.0, min(self.config_scroll, max_scroll))

    def _scroll_panel(self, mode: str, delta: float) -> None:
        if mode == "control":
            viewport = max(1.0, self.rect.height - 32)
            max_scroll = max(0.0, self.control_content_height - viewport)
            self.control_scroll = max(0.0, min(self.control_scroll + delta, max_scroll))
        else:
            viewport = max(1.0, self.rect.height - 32)
            max_scroll = max(0.0, self.config_content_height - viewport)
            self.config_scroll = max(0.0, min(self.config_scroll + delta, max_scroll))

    def _draw_message(self, surface: pg.Surface) -> None:
        if not self.message_text:
            return
        if pg.time.get_ticks() > self.message_until:
            self.message_text = None
            return
        msg = self.small_font.render(self.message_text, True, self.message_color)
        surface.blit(msg, (self.rect.x + 16, self.rect.bottom - 32))

    def _update_slider(self, name: str, mouse_x: int) -> None:
        rect = self.slider_rects.get(name)
        if not rect:
            return
        ratio = (mouse_x - rect.x) / rect.width
        ratio = max(0.0, min(1.0, ratio))
        controls.set_weight(name, ratio * 2.0)

    def _apply_count_action(self, kind: str, action: str) -> None:
        if kind == "robots":
            current = len(self.sim.robots)
            if action == "dec":
                self.sim.set_robot_count(max(0, current - 1))
            elif action == "inc":
                self.sim.set_robot_count(current + 1)
            elif action == "zero":
                self.sim.set_robot_count(0)
            label = "robots"
        else:
            current = len(self.sim.targets)
            if action == "dec":
                self.sim.set_target_count(max(0, current - 1))
            elif action == "inc":
                self.sim.set_target_count(current + 1)
            elif action == "zero":
                self.sim.set_target_count(0)
            label = "targets"
        self.notify(
            f"Set {label} = {len(self.sim.robots) if label == 'robots' else len(self.sim.targets)}",
            PANEL_ACCENT,
        )

    def _build_term_boxes(self) -> None:
        self.term_boxes = [
            TermBox(info, self.code_font) for _, info in controls.iter_terms()
        ]
        self._update_dirty_flag()

    def _save_term_boxes(self) -> None:
        entries = [box.to_entry() for box in self.term_boxes]
        success, message = controls.save_entries_to_json(entries)
        if not success:
            self.notify(message, MESSAGE_ERROR)
            return
        reload_ok, reload_msg = controls.reload_json_terms()
        if reload_ok:
            for box in self.term_boxes:
                box.mark_clean()
            self.config_dirty = False
            self._build_term_boxes()
            self.notify("Saved + applied JSON terms", MESSAGE_SUCCESS)
        else:
            self.notify(f"Saved JSON but reload failed: {reload_msg}", MESSAGE_ERROR)

    def _update_dirty_flag(self) -> None:
        self.config_dirty = any(box.is_dirty() for box in self.term_boxes)


class App:
    def __init__(self) -> None:
        pg.init()
        pg.display.set_caption("Live 2D Robot Sim")
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        self.clock = pg.time.Clock()
        self.running = True
        self.paused = False
        self.scene_path = SCENE_FILE
        if self.scene_path.exists():
            world, robots, obstacles, targets = load_scene(self.scene_path)
        else:
            world, robots, obstacles, targets = load_default_scene()
        self.sim = Simulator(
            world, robots, obstacles, targets, scene_path=self.scene_path
        )
        panel_rect = pg.Rect(WIDTH - PANEL_WIDTH, 0, PANEL_WIDTH, HEIGHT)
        font = pg.font.Font(None, 28)
        self.overlay = Overlay(panel_rect, font, self.sim)

    def run(self) -> None:
        accumulator = 0.0
        dt = 0.01
        target_fps = 60
        while self.running:
            elapsed = self.clock.tick(target_fps) / 1000.0
            accumulator += elapsed
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False
                self.overlay.handle_event(event)
                self.sim.handle_event(event)
                self._handle_key(event)
            while accumulator >= dt:
                if not self.paused:
                    self.sim.step(dt)
                accumulator -= dt
            self._draw()
        pg.quit()

    def _handle_key(self, event: pg.event.Event) -> None:
        if event.type != pg.KEYDOWN:
            return
        if self.overlay.handle_keydown(event):
            return
        ctrl = event.mod & pg.KMOD_CTRL
        if event.key == pg.K_SPACE:
            self.paused = not self.paused
        elif event.key == pg.K_r and not ctrl:
            self.sim.reset()
        elif event.key == pg.K_s:
            if ctrl:
                self.sim.save_scene(self.scene_path)
            else:
                self._save_snapshot()
        elif event.key == pg.K_l:
            if ctrl:
                self.sim.reset()
            else:
                self._write_log()
        elif event.key == pg.K_p:
            self._save_snapshot()
        elif event.key in TERM_KEYS:
            controls.toggle_term(TERM_KEYS[event.key])
        elif event.key == pg.K_a:
            self.sim.add_robot(np.array(pg.mouse.get_pos(), dtype=float))
        elif event.key == pg.K_g:
            self.sim.add_target(np.array(pg.mouse.get_pos(), dtype=float))
        elif event.key == pg.K_o:
            self.sim.add_obstacle(np.array(pg.mouse.get_pos(), dtype=float))
        elif event.key in (pg.K_d, pg.K_DELETE):
            self.sim.delete_selection()
        elif event.key == pg.K_TAB and ctrl:
            self.overlay.toggle_mode()

    def _draw(self) -> None:
        self.screen.fill(BACKGROUND)
        self.sim.draw(self.screen)
        fps = self.clock.get_fps()
        selection = self.sim.selection_summary()
        self.overlay.draw(self.screen, fps, self.paused, selection)
        pg.display.flip()

    def _save_snapshot(self) -> None:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = SNAPSHOT_DIR / f"snapshot_{timestamp}.png"
        pg.image.save(self.screen, filename)

    def _write_log(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = LOG_DIR / f"poses_{timestamp}.csv"
        rows = self.sim.pose_rows()
        with filename.open("w", encoding="utf-8") as handle:
            handle.write("robot_id,x,y\n")
            for idx, x, y in rows:
                handle.write(f"{idx},{x:.3f},{y:.3f}\n")


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()
