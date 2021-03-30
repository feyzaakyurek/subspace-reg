import pdb
import re

def main():
    
    file = "1shotnew.txt"
    with open(file, "r") as f:
        lines = [line.rstrip('\n') for line in f]

    
    base_query = []
    novel_support = []
    novel_query = []
    
    for i, line in enumerate(lines):
        if "Ekin: ids: " in line:
            s = lines[i] + " " + lines[i+1] + " " + lines[i+2]
            base_query.append(s[s.find("["):s.find("]")+1])

        if "Ekin: TrainIds: " in line:
            novel_support.append(line[line.find("[")+1:line.find("]")])
        if "Ekin: TestIds: " in line:
            novel_query.append(line[line.find("[")+1:line.find("]")])
    
    assert (5 * len(base_query)) == len(novel_support)
    assert (5 * len(base_query)) == len(novel_query)
    
    with open("/home/gridsan/akyurek/akyureklab_shared/rfs-incremental/data/miniImageNet/episodes_5_1.txt", "w") as f:
        f.write("%s\n" % "VAL_B")
        for _ in range(len(base_query)//2):
            f.write("\nBase Query: %s\n" % str(base_query.pop(0)))
            
            # Write the next 5 novel support entries
            f.write("Novel Support: [%s, %s, %s, %s, %s]\n" % (novel_support.pop(0),
                                                             novel_support.pop(0),
                                                             novel_support.pop(0),
                                                             novel_support.pop(0),
                                                             novel_support.pop(0)))
            # Write the next 5 novel support entries
            f.write("Novel Query: [%s, %s, %s, %s, %s]\n" % (novel_query.pop(0),
                                                             novel_query.pop(0),
                                                             novel_query.pop(0),
                                                             novel_query.pop(0),
                                                             novel_query.pop(0)))
            
        f.write("%s\n\n" % "TEST_B")
        for _ in range(len(base_query)//2):
            f.write("\nBase Query: %s\n" % str(base_query.pop(0)))
            
            # Write the next 5 novel support entries
            f.write("Novel Support: [%s, %s, %s, %s, %s]\n" % (novel_support.pop(0),
                                                             novel_support.pop(0),
                                                             novel_support.pop(0),
                                                             novel_support.pop(0),
                                                             novel_support.pop(0)))
            # Write the next 5 novel support entries
            f.write("Novel Query: [%s, %s, %s, %s, %s]\n" % (novel_query.pop(0),
                                                             novel_query.pop(0),
                                                             novel_query.pop(0),
                                                             novel_query.pop(0),
                                                             novel_query.pop(0)))
    
if __name__ == "__main__":
    main()