from multiprocessing import Pool,Process
import cProfile
def foo(a):
    return a
def main():
    
    inputs = [n for n in range(10000)]
    with Pool(8) as p:
        p.map(foo,inputs)
if __name__ == "__main__":
    cProfile.run("main()",sort = "time")