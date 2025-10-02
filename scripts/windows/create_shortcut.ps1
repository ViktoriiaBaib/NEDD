param(
  [Parameter(Mandatory=$true)][string]$ShortcutPath,
  [Parameter(Mandatory=$true)][string]$TargetPath,
  [string]$IconPath = "$env:SystemRoot\System32\SHELL32.dll",
  [string]$Description = "Launch app"
)

$wsh = New-Object -ComObject WScript.Shell
$sc = $wsh.CreateShortcut($ShortcutPath)
$sc.TargetPath = $TargetPath
$sc.WorkingDirectory = Split-Path -Path $TargetPath
$sc.WindowStyle = 1
$sc.IconLocation = $IconPath
$sc.Description = $Description
$sc.Save()
