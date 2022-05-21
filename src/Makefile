basic:
	chmod +x run.sh

le_branke: basic
	./run.sh le_branke
	python3 post_process.py -e ./le_branke/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

branin: basic
	./run.sh branin
	python3 post_process.py -e ./branin/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

levy: basic
	./run.sh levy
	python3 post_process.py -e ./levy/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

rosenbrock: basic
	./run.sh rosenbrock
	python3 post_process.py -e ./rosenbrock/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

ackley: basic
	./run.sh ackley
	python3 post_process.py -e ./ackley/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

hartmann: basic
	./run.sh hartmann
	python3 post_process.py -e ./hartmann/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret