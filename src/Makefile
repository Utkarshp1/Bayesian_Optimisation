EXPDIR=./$(shell date +%F)
ZACQ=EI
BUDGET=100
RUNS=20
NOISEMEAN=0.0
NOISEVAR=0.01
NUMRESTARTS=10
RAWSAMPLES=32
RANDOMSEED=42
INIT_EXAMPLES=5
ARGS=$(EXPDIR) $(ZACQ) $(BUDGET) $(RUNS) $(NOISEMEAN) $(NOISEVAR) $(NUMRESTARTS) $(RAWSAMPLES) $(RANDOMSEED) $(INIT_EXAMPLES)

basic:
	chmod +x run.sh

le_branke: basic
	./run.sh le_branke max 1 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/le_branke/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

branin: basic
	./run.sh branin min 2 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/branin/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

levy: basic
	./run.sh levy min 4 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/levy/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

rosenbrock: basic
	./run.sh rosenbrock min 4 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/rosenbrock/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

ackley: basic
	./run.sh ackley min 5 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/ackley/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

hartmann: basic
	./run.sh hartmann min 6 $(ARGS)
	python3 post_process.py -e $(EXPDIR)/hartmann/ -s 0.1 -x Number\ of\ Iterations -y Mean\ Regret

illustration6d: basic
	./run.sh illustrationNd min 6 $(ARGS)

illustration10d: basic
	./run.sh illustrationNd min 10 $(ARGS)