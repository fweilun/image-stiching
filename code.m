function []=photo_stitching()

    %讀取圖片
    I1 = imread("1.jpg");
    I2 = imread("2.jpg");
    image1Size=size(I1);
    image2Size=size(I2);
    imshow([I1,I2])
    title( "原始圖" )
    
    % 圖片灰化&求亮度總和
    Image1 = rgb2gray(I1);
    Image2 = rgb2gray(I2);
    sum_1 = sum(im2double(Image1), "all");
    sum_2 = sum(im2double(Image2), "all");
    exposure_value1 = sum_1 / (image1Size(1)*image1Size(2));
    exposure_value2 = sum_2 / (image2Size(1)*image2Size(2));
    mean_expo_val = (exposure_value1+exposure_value2) / 2;

    %抓出兩張圖透過sift找到的特徵點
    points1 = detectSIFTFeatures(Image1);
    points2 = detectSIFTFeatures(Image2);

    %特徵點配對
    [Image1Features, points1] = extractFeatures(Image1, points1);
    [Image2Features, points2] = extractFeatures(Image2, points2);
    boxPairs = matchFeatures(Image1Features, Image2Features);
    matchedimg1Points = points1(boxPairs(:, 1));
    matchedimg2Points = points2(boxPairs(:, 2));
    figure();
    showMatchedFeatures(I1,I2,matchedimg1Points,matchedimg2Points,'montage')
    title( "未使用ransac之配對" )

    %透過RANSAC算法刪除異常值
    [tform,inlierIdx] = estgeotform2d(matchedimg1Points,matchedimg2Points, "projective" );
    inlierimg1Points = matchedimg1Points(inlierIdx,:);
    inlierimg2Points = matchedimg2Points(inlierIdx,:);
    figure()
    showMatchedFeatures(I1,I2,inlierimg1Points,inlierimg2Points,'montage')
    title( "使用ransac後匹配的內點" )

    %透過得到的homography matrix將I1轉至I2上
    Rfixed = imref2d(size(I2));

    [registered1, Rregistered1] = imwarp(I1, tform);
    figure()
    imshowpair(I2,Rfixed,registered1,Rregistered1,"blend","Scaling","joint");
    title('拼接上的圖片');
    
    %儲存世界座標
    WL = [Rregistered1.XWorldLimits; Rfixed.XWorldLimits];
    WL(:,:,2) = [Rregistered1.YWorldLimits;Rfixed.YWorldLimits];%(I1 I2, start end, x y)
    
    %對世界座標四捨五入
    for i=1:2
        for j=1:2
            for k=1:2
                if(WL(i,j,k) < 0 && mod(WL(i,j,k), 1) == 0.5)
                    WL(i,j,k) = round(WL(i,j,k)) + 1;
                    continue;
                end
                WL(i,j,k) = round(WL(i,j,k));
            end
        end
    end
    
    %將世界座標平移為正數
    tmp = min(WL,[],1);
    tmp = min(tmp,[],2);
    WL = WL(:,:,:) - tmp;

    tmp = max(WL,[],1);
    tmp = max(tmp,[],2);
    rf = tmp(1,1,2);
    cf = tmp(1,1,1);
    final = zeros(rf, cf, 3);
    coo_I1 = WL(1,:,:);
    m_1 = im2double(registered1);
    m_1 = m_1*mean_expo_val/exposure_value1;

    for i = 1 : coo_I1(1,2,2) - coo_I1(1,1,2)
        for j = 1 : coo_I1(1,2,1) - coo_I1(1,1,1)
            final(coo_I1(1,1,2)+i, coo_I1(1,1,1)+j, :) = m_1(i, j, :);
        end
    end

    coo_I2 = WL(2,:,:);
    m_2 = im2double(I2);
    m_2 = m_2*mean_expo_val/exposure_value2;

    bound_m1 = zeros(image1Size);
    bound_m1(1,:) = 1;  bound_m1(image1Size(1),:) = 1;
    bound_m1(:,1) = 1;  bound_m1(:,image1Size(2)) = 1;
    trans_bound_m1 = imwarp(bound_m1, tform);
    bound_dist1 = bwdist(trans_bound_m1);

    bound_m2 = zeros(size(Image2));
    bound_m2(1,:) = 1;  bound_m2(image1Size(1),:) = 1;
    bound_m2(:,1) = 1;  bound_m2(:,image1Size(2)) = 1;
    bound_dist2 = bwdist(bound_m2);

    for i = 1 : coo_I2(1,2,2) - coo_I2(1,1,2)
        for j = 1 : coo_I2(1,2,1) - coo_I2(1,1,1)
            if final(coo_I2(1,1,2)+i, coo_I2(1,1,1)+j, :) == [0, 0, 0]
                final(coo_I2(1,1,2)+i, coo_I2(1,1,1)+j, :) = m_2(i, j, :);
            else
                i_I1 = coo_I2(1,1,2)+i-coo_I1(1,1,2);
                j_I1 = coo_I2(1,1,1)+j-coo_I1(1,1,1);
                w_1 = bound_dist1(i_I1, j_I1);
                w_2 = bound_dist2(i, j);

                final(coo_I2(1,1,2)+i, coo_I2(1,1,1)+j, :) = (final(coo_I2(1,1,2)+i, coo_I2(1,1,1)+j, :)*w_1 + m_2(i, j, :)*w_2) / (w_1+w_2);
            end
        end
    end

    %圖片輸出
    final_photo = uint8(final*255);
    figure();
    imshow(final_photo);
end
