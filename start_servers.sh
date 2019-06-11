#!/bin/sh

# In gnome-terminal preferences, create a new profile named "Hold".
# Go to command tab, select "Hold the terminal" under "When command exists" option.
terminal=gnome-terminal;

run_django_server_loop()
{
  command="python3 manage.py runserver";
  until $command; do
      echo "Server crashed with exit code $?.  Respawning.." >&2
      sleep 0.5
  done
}

echo "Starting Django server";
export -f run_django_server_loop
$terminal --window-with-profile=Hold -x bash -c 'run_django_server_loop; bash'
cd interface
npm run dev
