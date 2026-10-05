@echo off
rem One command to (re)run the whole HD texture pipeline. Resumable: just run it again.
rem Extra arguments are passed through, e.g.  hdtex.bat --limit 50
python "%~dp0hdtex.py" all %*
