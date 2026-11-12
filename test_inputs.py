import ctypes
from ctypes import wintypes
import sys
import time

from src.actuator import (
    PureKeyboardActuator,
    VK_LEFT,
    VK_RIGHT,
    VK_SPACE,
    VK_A,
    VK_D,
    ensure_desktop_access,
)

user32 = ctypes.windll.user32

def find_game_window() -> int:
    ensure_desktop_access()
    import win32gui
    import win32process
    import psutil

    found_hwnd = 0

    def enum_cb(hwnd, _):
        nonlocal found_hwnd
        if win32gui.IsWindowVisible(hwnd):
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                pname = psutil.Process(pid).name().lower()
                if pname in ("ci1.exe", "chickeninvaders.exe", "chicken invaders.exe"):
                    found_hwnd = hwnd
                    return False
            except Exception:
                pass

            title = win32gui.GetWindowText(hwnd).strip()
            if title in ("Chicken Invaders 1", "Chicken Invaders"):
                found_hwnd = hwnd
                return False
        return True

    try:
        win32gui.EnumWindows(enum_cb, None)
    except Exception:
        pass
    return found_hwnd

def focus_game_window(hwnd: int):
    import win32con
    import win32gui
    import win32process

    try:
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            time.sleep(0.05)

        kernel32 = ctypes.windll.kernel32
        current_thread = kernel32.GetCurrentThreadId()
        target_thread, _ = win32process.GetWindowThreadProcessId(hwnd)
        fg_hwnd = user32.GetForegroundWindow()
        fg_thread = win32process.GetWindowThreadProcessId(fg_hwnd)[0] if fg_hwnd else 0

        if fg_thread and fg_thread != current_thread:
            user32.AttachThreadInput(current_thread, fg_thread, True)
        if target_thread and target_thread != current_thread:
            user32.AttachThreadInput(current_thread, target_thread, True)

        user32.ShowWindow(hwnd, win32con.SW_SHOW)
        user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)
        user32.SetWindowPos(hwnd, -2, 0, 0, 0, 0, 0x0001 | 0x0002)
        user32.SwitchToThisWindow(hwnd, True)
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)

        if fg_thread and fg_thread != current_thread:
            user32.AttachThreadInput(current_thread, fg_thread, False)
        if target_thread and target_thread != current_thread:
            user32.AttachThreadInput(current_thread, target_thread, False)

        time.sleep(0.2)
    except Exception as e:
        print(f"Error focusing window: {e}")

def main():
    print("=" * 65)
    print("  Chicken Invaders - Input Diagnostic Test Tool")
    print("=" * 65)

    ensure_desktop_access()
    hwnd = find_game_window()

    if not hwnd:
        print("\n[!] Chicken Invaders is NOT currently running or visible on the screen.")
        print("\nInstructions:")
        print("  1. Launch Chicken Invaders 1 on your computer (start CI1.exe).")
        print("  2. Start a game (press Fire / Space until you are on Wave 1).")
        print("  3. Run this script again: python test_inputs.py")
        return

    import win32gui
    title = win32gui.GetWindowText(hwnd)
    print(f"\n[+] Detected Game Window: '{title}' (HWND: {hwnd})")
    print("[*] Bringing Chicken Invaders to foreground...")
    focus_game_window(hwnd)

    actuator = PureKeyboardActuator(keyset="arrows")

    print("\n-------------------------------------------------------------")
    print("Get ready! Testing inputs in 3 seconds. WATCH YOUR SHIP!")
    print("-------------------------------------------------------------")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1.0)

    # TEST 1: Fire Lasers
    print("\n[TEST 1] Firing 3 Lasers (Spacebar pulse)...")
    for i in range(3):
        print(f"  -> Fire pulse #{i+1}")
        actuator.fire_pulse(duration=0.06)
        time.sleep(0.20)

    time.sleep(0.5)

    # TEST 2: Discrete Bashes / Pulses (Left Arrow)
    print("\n[TEST 2] Discrete Movement Bashes - LEFT (5 pulses of 60ms)...")
    for i in range(5):
        print(f"  -> Left pulse #{i+1}")
        actuator.pulse_left(duration=0.06)
        time.sleep(0.08)

    time.sleep(0.5)

    # TEST 3: Discrete Bashes / Pulses (Right Arrow)
    print("\n[TEST 3] Discrete Movement Bashes - RIGHT (5 pulses of 60ms)...")
    for i in range(5):
        print(f"  -> Right pulse #{i+1}")
        actuator.pulse_right(duration=0.06)
        time.sleep(0.08)

    time.sleep(0.5)

    # TEST 4: Continuous Pulsed Hold (Left Arrow)
    print("\n[TEST 4] Continuous Pulsed Hold - LEFT (1.5 seconds)...")
    t_end = time.time() + 1.5
    while time.time() < t_end:
        actuator.press_left()
        time.sleep(0.03)
    actuator.release_movement()

    time.sleep(0.5)

    # TEST 5: Continuous Pulsed Hold (Right Arrow)
    print("\n[TEST 5] Continuous Pulsed Hold - RIGHT (1.5 seconds)...")
    t_end = time.time() + 1.5
    while time.time() < t_end:
        actuator.press_right()
        time.sleep(0.03)
    actuator.release_movement()

    # TEST 6: Alternative WASD Keys (A and D)
    print("\n[TEST 6] Testing WASD fallback (A = Left, D = Right)...")
    wasd_actuator = PureKeyboardActuator(keyset="wasd")
    print("  -> Pulse A (Left)")
    wasd_actuator.pulse_left(duration=0.10)
    time.sleep(0.3)
    print("  -> Pulse D (Right)")
    wasd_actuator.pulse_right(duration=0.10)
    time.sleep(0.3)

    print("\n" + "=" * 65)
    print("  Test Run Complete!")
    print("=" * 65)
    print("Summary of actions performed:")
    print("  1. Fired 3 laser pulses")
    print("  2. Moved Left via 5 discrete pulses (bashes)")
    print("  3. Moved Right via 5 discrete pulses (bashes)")
    print("  4. Held Left for 1.5s with continuous hardware repeats")
    print("  5. Held Right for 1.5s with continuous hardware repeats")
    print("  6. Tested A and D key pulses")
    print("-------------------------------------------------------------")

if __name__ == "__main__":
    main()
