# 捕获指定标题窗口的截图（用于 Viewer 取证）。
# 用法: powershell -ExecutionPolicy Bypass -File capture_window.ps1 -TitleLike "MuJoCo*" -Out "D:\...\shot.png"
param(
  [string]$TitleLike = "MuJoCo*",
  [string]$Out = "screenshot.png",
  [int]$WaitSeconds = 0
)

Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class NativeWin {
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr hdc, uint flags);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
  public struct RECT { public int Left, Top, Right, Bottom; }
}
"@

if ($WaitSeconds -gt 0) { Start-Sleep -Seconds $WaitSeconds }

$proc = Get-Process python, pythonw -ErrorAction SilentlyContinue |
        Where-Object { $_.MainWindowTitle -like $TitleLike } |
        Select-Object -First 1
if (-not $proc) { Write-Output "NO_WINDOW"; exit 2 }

$h = $proc.MainWindowHandle
[void][NativeWin]::ShowWindow($h, 9)
[void][NativeWin]::SetForegroundWindow($h)
Start-Sleep -Milliseconds 1200

$r = New-Object NativeWin+RECT
[void][NativeWin]::GetWindowRect($h, [ref]$r)
$w = $r.Right - $r.Left
$hgt = $r.Bottom - $r.Top
if ($w -le 0 -or $hgt -le 0) { Write-Output "BAD_RECT"; exit 3 }

$bmp = New-Object System.Drawing.Bitmap $w, $hgt
$g = [System.Drawing.Graphics]::FromImage($bmp)
$hdc = $g.GetHdc()
$ok = [NativeWin]::PrintWindow($h, $hdc, 2)
$g.ReleaseHdc($hdc)
$g.Dispose()
$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$bmp.Dispose()

Write-Output ("CAPTURED title='{0}' size={1}x{2} printwindow={3} -> {4}" -f $proc.MainWindowTitle, $w, $hgt, $ok, $Out)
