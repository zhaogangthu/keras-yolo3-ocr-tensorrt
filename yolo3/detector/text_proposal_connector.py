import numpy as np
from yolo3.detector.text_proposal_graph_builder import TextProposalGraphBuilder

class TextProposalConnector:
    """
        Connect text proposals into text lines
    """
    def __init__(self,MAX_HORIZONTAL_GAP=30,MIN_V_OVERLAPS=0.6,MIN_SIZE_SIM=0.6):
        self.graph_builder=TextProposalGraphBuilder(MAX_HORIZONTAL_GAP,MIN_V_OVERLAPS,MIN_SIZE_SIM)

    def group_text_proposals(self, text_proposals, scores, im_size):
        graph=self.graph_builder.build_graph(text_proposals, scores, im_size)
        return graph.sub_graphs_connected()

    def fit_y(self, X, Y, x1, x2):
        len(X)!=0
        # if X only include one point, the function will get line y=Y[0]
        if np.sum(X==X[0])==len(X):
            return Y[0], Y[0]
        p=np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores, im_size):
        """
        text_proposals:boxes
        
        """
        # tp=text proposal
        tp_groups=self.group_text_proposals(text_proposals, scores, im_size)##find the text line 
        
        text_lines=np.zeros((len(tp_groups), 8), np.float32)

        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes=text_proposals[list(tp_indices)]
            #num = np.size(text_line_boxes)##find 
            X = (text_line_boxes[:,0] + text_line_boxes[:,2]) / 2
            Y = (text_line_boxes[:,1] + text_line_boxes[:,3]) / 2
            
            z1 = np.polyfit(X,Y,1)
           # p1 = np.poly1d(z1)


            x0=np.min(text_line_boxes[:, 0])
            x1=np.max(text_line_boxes[:, 2])

            offset=(text_line_boxes[0, 2]-text_line_boxes[0, 0])*0.5

            lt_y, rt_y=self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0+offset, x1-offset)
            lb_y, rb_y=self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0+offset, x1-offset)

            # the score of a text line is the average score of the scores
            # of all text proposals contained in the text line
            score=scores[list(tp_indices)].sum()/float(len(tp_indices))

            text_lines[index, 0]=x0
            text_lines[index, 1]=min(lt_y, rt_y)
            text_lines[index, 2]=x1
            text_lines[index, 3]=max(lb_y, rb_y)
            text_lines[index, 4]=score
            text_lines[index, 5]=z1[0]
            text_lines[index, 6]=z1[1]
            height = np.mean( (text_line_boxes[:,3]-text_line_boxes[:,1]) )
            text_lines[index, 7]= height + 2.5
        #text_lines=clip_boxes(text_lines, im_size)

        # text_recs = np.zeros((len(text_lines), 9), np.float)
        # index = 0
        # for line in text_lines:
        #     b1 = line[6] - line[7] / 2  # 根据高度和文本行中心线，求取文本行上下两条线的b值
        #     b2 = line[6] + line[7] / 2
        #     x1 = line[0]
        #     y1 = line[5] * line[0] + b1  # 左上
        #     x2 = line[2]
        #     y2 = line[5] * line[2] + b1  # 右上
        #     x3 = line[0]
        #     y3 = line[5] * line[0] + b2  # 左下
        #     x4 = line[2]
        #     y4 = line[5] * line[2] + b2  # 右下
        #     disX = x2 - x1
        #     disY = y2 - y1
        #     width = np.sqrt(disX * disX + disY * disY)  # 文本行宽度
        #
        #     fTmp0 = y3 - y1  # 文本行高度
        #     fTmp1 = fTmp0 * disY / width
        #     x = np.fabs(fTmp1 * disX / width)  # 做补偿
        #     y = np.fabs(fTmp1 * disY / width)
        #     if line[5] < 0:
        #         x1 -= x
        #         y1 += y
        #         x4 += x
        #         y4 -= y
        #     else:
        #         x2 += x
        #         y2 += y
        #         x3 -= x
        #         y3 -= y
        #     text_recs[index, 0] = x1
        #     text_recs[index, 1] = y1
        #     text_recs[index, 2] = x2
        #     text_recs[index, 3] = y2
        #     text_recs[index, 4] = x3
        #     text_recs[index, 5] = y3
        #     text_recs[index, 6] = x4
        #     text_recs[index, 7] = y4
        #     text_recs[index, 8] = line[4]
        #     index = index + 1


        return text_lines
