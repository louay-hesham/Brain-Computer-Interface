#!/bin/sh

terminal=gnome-terminal;

# In gnome-terminal preferences, create a new profile named "Hold".
# Go to command tab, select "Hold the terminal" under "When command exists" option.

echo "Starting Django server";
$terminal --window-with-profile=Hold -- python3 manage.py runserver;

# Run frontend server below
