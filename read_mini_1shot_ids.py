import pdb

def main():
    
    file = "printids_1sot.out"
    with open(file, "r") as f:
        lines = [line.rstrip('\n') for line in f]

    
    base_query = []
    novel_support = []
    novel_query = []
    
    for line in lines:
        if "Base Query: " in line:
            base_query.append(line.split(" ", 2)[2])
        if "Novel Support: " in line:
            novel_support.append(int(line.split(" ", 2)[2]))
        if "Novel Query: " in line:
            novel_query.append(line.split(" ", 2)[2])
#         pdb.set_trace()    
    pdb.set_trace()    
    assert (5 * len(base_query)) == len(novel_support)
    assert (5 * len(base_query)) == len(novel_query)
    
    with open("~/akyureklab_shared/rfs-incremental/data/miniImageNet/episodes_5_1.txt", "w") as f:
        f.write("%s\n\n" % "VAL_B")
        for _ in range(len(base_query)//2):
            f.write("Base Query: [%s]\n" % str(base_query.pop(0)))
            
            # Write the next 5 novel support entries
            f.write("Novel Support: [%d, %d, %d, %d, %d]\n" % (novel_support.pop(0),
                                                             novel_support.pop(0),
                                                             novel_support.pop(0),
                                                             novel_support.pop(0),
                                                             novel_support.pop(0)))
            # Write the next 5 novel support entries
            f.write("Novel Query: [%s, %s, %s, %s, %s]\n" % (novel_query.pop(0).replace(" ", ", "),
                                                             novel_query.pop(0).replace(" ", ", "),
                                                             novel_query.pop(0).replace(" ", ", "),
                                                             novel_query.pop(0).replace(" ", ", "),
                                                             novel_query.pop(0).replace(" ", ", ")))
            
        f.write("%s\n\n" % "TEST_B")
        for _ in range(len(base_query)//2):
            f.write("Base Query: %s\n" % str(base_query.pop(0)))
            
            # Write the next 5 novel support entries
            f.write("Novel Support: [%d, %d, %d, %d, %d]\n" % (novel_support.pop(0),
                                                             novel_support.pop(0),
                                                             novel_support.pop(0),
                                                             novel_support.pop(0),
                                                             novel_support.pop(0)))
            # Write the next 5 novel support entries
            f.write("Novel Query: [%s, %s, %s, %s, %s]\n" % (novel_query.pop(0).replace(" ", ", "),
                                                             novel_query.pop(0).replace(" ", ", "),
                                                             novel_query.pop(0).replace(" ", ", "),
                                                             novel_query.pop(0).replace(" ", ", "),
                                                             novel_query.pop(0).replace(" ", ", ")))
    
if __name__ == "__main__":
    main()