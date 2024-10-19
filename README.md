# Tagmane (プレリリース版)

![Tagmane Cover](asset/cover.png)

Tagmaneは、画像のタグ管理を効率化するためのデスクトップアプリケーションです。機械学習モデルを使用して自動的にタグを生成し、ユーザーが手動でタグを編集・管理することができます。

**注意: 現在のバージョンはプレリリース版です。機能が不完全であったり、予期せぬ動作をする可能性があります。**

## 主な機能

1. **フォルダ選択**: 画像が含まれるフォルダを選択し、一括で画像を読み込みます。

2. **タグの自動生成**: VLM（Vision Language Model）を使用して、画像に適したタグを自動的に生成します。

3. **タグの手動編集**: 
   - タグの追加、削除、並び替えが可能です。
   - ドラッグ＆ドロップでタグの順序を変更できます。

4. **フィルタリング機能**: 
   - 特定のタグを含む画像のみを表示できます。
   - ANDモードとORモードを切り替えて、複数のタグでフィルタリングできます。

5. **Undo/Redo機能**: タグの編集操作を取り消したり、やり直したりすることができます。

6. **タグの一括操作**: 選択したタグをすべての画像から一括で削除することができます。

7. **タグの保存**: 編集したタグを各画像に対応するテキストファイルに保存できます。

8. **ログ機能**: 操作ログやデバッグログを表示し、アプリケーションの動作を確認できます。

## 使用方法

1. アプリケーションを起動し、「フォルダを選択」ボタンをクリックして画像フォルダを選択します。
2. 画像リストから画像を選択すると、中央に画像が表示されます。
3. 「VLM推論」ボタンをクリックすると、選択した画像に対してタグが自動生成されます。
4. 右側のパネルでタグを編集します。タグの追加、削除、並び替えが可能です。
5. フィルタリングボタンを使用して、特定のタグを含む画像のみを表示できます。
6. 編集が完了したら、「タグを保存」ボタンをクリックしてタグをファイルに保存します。

## 要件

- .NET Framework 4.7.2以上
- Windows 7以上

## インストール

現在、Tagmaneはデバッグモードでのみ実行可能です。以下の手順で実行してください：

1. Visual Studioでソリューションファイル（.sln）を開きます。
2. デバッグモードで実行します（F5キーを押すか、「デバッグ」メニューから「デバッグの開始」を選択）。

注意：現在のバージョンでは、実行可能ファイル（.exe）での配布は行っていません。

## ライセンス

このプロジェクトはApache License 2.0の下で公開されています。詳細については、[LICENSE](LICENSE)ファイルを参照してください。

**注意**: 今後、ライセンスが変更される可能性があります。本ソフトウェアを使用することで、将来的なライセンス変更の可能性に同意したものとみなされます。

## 貢献

バグ報告や機能リクエストは、GitHubのIssueを通じて行ってください。プルリクエストも歓迎します。

## 作者

cella

## 免責事項

このソフトウェアはプレリリース版であり、開発中の状態です。使用する際は自己責任でお願いします。開発者は、このソフトウェアの使用によって生じたいかなる損害についても責任を負いません。
