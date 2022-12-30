function [ accuracy ] = calculateAccuracy( predLabel, trueLabel )
    dataNum = size(trueLabel,1);
    rightNum = 0;
    for i = 1:dataNum
        if predLabel(i,:) == trueLabel(i,:)
            rightNum = rightNum + 1;
        end
    end
    accuracy = rightNum/dataNum;
end

