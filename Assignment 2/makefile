install: venv
	. venv/bin/activate; pip3 install -Ur requirements.txt
venv :
	test -d venv || python3 -m venv venv

run1:
	. venv/bin/activate; python3 Scenario1.py

run2:
	. venv/bin/activate; python3 Scenario2.py

run3:
	. venv/bin/activate; python3 Scenario3.py

clean:
	rm -rf venv
	rm *.png