import geoip2.database
import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_coordinates(csv_file_path):
    # 事前に定義するファイルパス
    db_path = 'GeoLite2-City.mmdb'  # GeoLite2のデータベースパス
    output_csv_path = 'latlng_output.csv'  # 緯度経度出力用CSVパス
    output_plot_path = 'coordinates_plot.png'  # 出力画像のパス

    try:
        # 1. GeoLite2データベースを読み込む
        reader = geoip2.database.Reader(db_path)
    except FileNotFoundError:
        print("GeoLite2-City.mmdb のファイルが見つかりません。正しいパスを指定してください。")
        return

    try:
        # 2. IPアドレスリストをCSVから読み込む
        data = pd.read_csv(csv_file_path, header=None, names=['ip_address', 'value'])
    except Exception as e:
        print(f"CSVファイルの読み込みに失敗しました: {e}")
        return

    # 3. 緯度・経度を取得し、リストに保存する
    locations = []
    output_data = []  # IPアドレス、緯度、経度を保存するリスト

    for ip in data['ip_address'].drop_duplicates():  # 重複IPの除去
        try:
            response = reader.city(ip)
            lat = response.location.latitude
            lon = response.location.longitude
            if lat is not None and lon is not None:
                locations.append([lat, lon])
                output_data.append([ip, lat, lon])
        except geoip2.errors.AddressNotFoundError:
            pass  # 位置情報が見つからない場合は無視
        except Exception as e:
            print(f"IP: {ip} の取得中にエラーが発生しました - {e}")

    # データベースを閉じる
    reader.close()

    # 4. データをCSVファイルに保存
    if output_data:
        df = pd.DataFrame(output_data, columns=['IP Address', 'Latitude', 'Longitude'])
        df.to_csv(output_csv_path, index=False)
        print(f"IPアドレスと緯度経度が '{output_csv_path}' に保存されました。")
    else:
        print("取得できたIPの位置情報がありませんでした。")

    # 5. 2次元散布図を作成する
    if locations:
        latitudes = [loc[0] for loc in locations]
        longitudes = [loc[1] for loc in locations]

        plt.figure(figsize=(10, 6))
        plt.scatter(longitudes, latitudes, c='blue', marker='o', alpha=0.6)
        plt.title('IP Address Geolocation (Latitude vs Longitude)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)

        # 6. グラフを画像ファイルとして保存
        plt.savefig(output_plot_path)
        print(f"2次元座標プロットが '{output_plot_path}' に保存されました。")

        # 7. プロットを表示
        plt.show()
    else:
        print("プロットするデータがありません。")

if __name__ == "__main__":
    # コマンドライン引数からCSVファイル名を取得
    if len(sys.argv) < 2:
        print("使用方法: python script_name.py <CSVファイル名>")
        sys.exit(1)

    csv_file_path = sys.argv[1]

    # 関数を実行
    plot_coordinates(csv_file_path)